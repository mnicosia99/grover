import os
import json
import requests
import tensorflow as tf
import numpy as np
import sys
import feedparser
import time
from datetime import datetime, timedelta
import requests
import base64
from fakedata import create_authors, get_random_school, get_random_department, get_date
from create_pdf import write_to_pdf
from ttp import ttp

sys.path.append('../')
from ../lm.modeling import GroverConfig, sample
from ../sample.encoder import get_encoder, _tokenize_article_pieces, extract_generated_target
import random

# Fake person who will be slandered/libeled in the fake articles
NAME_TO_SLANDER = "Chucky McChuckster"
# IMAGE_TO_SLANDER = "https://images.generated.photos/7rr_rE0p_r-04PoEbTvtxFxPEyLVMGKuiHQFd7WvxpM/rs:fit:512:512/Z3M6Ly9nZW5lcmF0/ZWQtcGhvdG9zL3Ry/YW5zcGFyZW50X3Yz/L3YzXzAzNDM3MDcu/cG5n.png"

# SLANDEROUS_SEED_HEADLINES = [
#   f"{NAME_TO_SLANDER} convicted of stealing puppies",
#   f"{NAME_TO_SLANDER} caught lying about growing the world's largest watermelon",
#   f"{NAME_TO_SLANDER} single-handedly started the Cold War",
#   f"{NAME_TO_SLANDER} forged priceless works of modern art for decades",
#   f"{NAME_TO_SLANDER} claimed to be Pokemon master, but caught in a lie",
#   f"{NAME_TO_SLANDER} bought fake twitter followers to pretend to be a celebrity",
#   f"{NAME_TO_SLANDER} created the original design for Ultron",
#   f"{NAME_TO_SLANDER} revealed as a foriegn spy for the undersea city of Atlantis",
#   f"{NAME_TO_SLANDER} involved in blackmail scandal with King Trident of Atlantis",
#   f"{NAME_TO_SLANDER} is dumb",
#   f"{NAME_TO_SLANDER} lied on tax returns to cover up past life as a Ninja Turtle",
#   f"{NAME_TO_SLANDER} stole billions from investors in a new pet store",
#   f"{NAME_TO_SLANDER} claims to be a Ninja Turtle but was actually lying",
#   f"{NAME_TO_SLANDER} likely to be sentenced to 20 years in jail for chasing a cat into a tree",
#   f"{NAME_TO_SLANDER} caught in the act of illegal trafficking of Teletubbies",
#   f"{NAME_TO_SLANDER} commits a multitude of crimes against dinosaurs",
# ]

# def get_fake_articles(domain):
#     articles = []

#     headlines_to_inject = SLANDEROUS_SEED_HEADLINES

#     for fake_headline in headlines_to_inject:
#         days_ago = random.randint(1, 7)
#         pub_datetime = datetime.now() - timedelta(days=days_ago)

#         publish_date = pub_datetime.strftime('%m-%d-%Y')
#         iso_date = pub_datetime.isoformat()

#         articles.append({
#             'summary': "",
#             'title': fake_headline,
#             'text': '',
#             'authors': ["Staff Writer"],
#             'publish_date': publish_date,
#             'iso_date': iso_date,
#             'domain': domain,
#             'image_url': IMAGE_TO_SLANDER,
#             'tags': ['Breaking News', 'Investigations', 'Criminal Profiles'],
#         })

#     return articles

# def get_articles_from_real_blog(domain, feed_url):
#     feed_data = feedparser.parse(feed_url)
#     articles = []
#     for post in feed_data.entries:
#         if 'published_parsed' in post:
#             publish_date = time.strftime('%m-%d-%Y', post.published_parsed)
#             iso_date = datetime(*post.published_parsed[:6]).isoformat()
#         else:
#             publish_date = time.strftime('%m-%d-%Y')
#             iso_date = datetime.now().isoformat()

#         if 'summary' in post:
#             summary = post.summary
#         else:
#             summary = None

#         tags = []
#         if 'tags' in post:
#             tags = [tag['term'] for tag in post['tags']]
#             if summary is None:
#                 summary = ", ".join(tags)

#         image_url = None
#         if 'media_content' in post:
#             images = post.media_content
#             if len(images) > 0 and 'url' in images[0]:
#                 image_url = images[0]['url']
#                 # Hack for NYT images to fix tiny images in the RSS feed
#                 if "-moth" in image_url:
#                     image_url = image_url.replace("-moth", "-threeByTwoMediumAt2X")

#         if 'authors' in post:
#             authors = list(map(lambda x: x["name"], post.authors))
#         else:
#             authors = ["Staff Writer"]

#         articles.append({
#             'summary': summary,
#             'title': post.title,
#             'text': '',
#             'authors': authors,
#             'publish_date': publish_date,
#             'iso_date': iso_date,
#             'domain': domain,
#             'image_url': image_url,
#             'tags': tags,
#         })

#     return articles
def generate_html_from_article_text(article_title, article_text, authors, publish_date, university, department, image_url=None):
    
    # parse the article text
    p = ttp.Parser()
    result = p.parse(article_text)
    article_text = result.html
    
    #  split article on newlines
    lines = article_text.split("\n")
   
    new_lines = []
    for line in lines:
        new_lines.append(line)
    
    #  add paragraph between each new line
    article_text = "<p>".join(new_lines)    
    
    # if an image url exists, add it to the returned html
    if image_url is not None:
        article_text = "<img src=" + image_url + "/>" + article_text
    
    #  create html from article text
    style = " hr.solid { border-top: 3px solid #bbb;}"
    article_title = "<h1>" + article_title + "</h1>\n"
    authors = "<p>Authors: " + authors + "</p>\n"
    publish_date = "Published Date: " + publish_date + "<br/>\n"
    university = "<p>" + university + "</p>\n"
    department = department + "<br/>\n"
    sep = "<hr class=\"solid\">\n"
    body = article_title + authors + publish_date + sep + university + department + sep + article_text
    article_html = "<html><head><style>" + style + "</style></head><body>" + body + "</body></html>"
 
    # return the article as html
    return article_html

def generate_article_attribute(sess, encoder, tokens, probs, article, target='article'):

    # Tokenize the raw article text from sample grover encoder
    article_pieces = _tokenize_article_pieces(encoder, article)

    # Grab the article elements the model careas about - domain, date, title, etc.
    context_formatted = []
    for key in ['domain', 'date', 'authors', 'title', 'article']:
        if key != target:
            context_formatted.extend(article_pieces.pop(key, []))

    # Start formatting the tokens in the way the model expects them, starting with
    # which article attribute we want to generate.
    context_formatted.append(encoder.__dict__['begin_{}'.format(target)])
    # Tell the model which special tokens (such as the end token) aren't part of the text
    ignore_ids_np = np.array(encoder.special_tokens_onehot)
    ignore_ids_np[encoder.__dict__['end_{}'.format(target)]] = 0

    # We are only going to generate one article attribute with a fixed
    # top_ps cut-off of 95%. This simple example isn't processing in batches.
    gens = []
    article['top_ps'] = [0.95]

    # Run the input through the TensorFlow model and grab the generated output
    tokens_out, probs_out = sess.run(
        [tokens, probs],
        feed_dict={
            # Pass real values for the inputs that the
            # model needs to be able to run.
            initial_context: [context_formatted],
            eos_token: encoder.__dict__['end_{}'.format(target)],
            ignore_ids: ignore_ids_np,
            p_for_topp: np.array([0.95]),
        }
    )

    # The model is done! Grab the results it generated and format the results into normal text.
    for t_i, p_i in zip(tokens_out, probs_out):
        extraction = extract_generated_target(output_tokens=t_i, encoder=encoder, target=target)
        gens.append(extraction['extraction'])

    # Return the generated text.
    return gens[-1]

model_type = "mega"

model_dir = os.path.join('/content/gdrive/MyDrive/grover-fork2/grover/models', model_type)
if not os.path.exists(model_dir):
    os.makedirs(model_dir)

for ext in ['data-00000-of-00001', 'index', 'meta']:
    r = requests.get(f'https://storage.googleapis.com/grover-models/{model_type}/model.ckpt.{ext}', stream=True)
    with open(os.path.join(model_dir, f'model.ckpt.{ext}'), 'wb') as f:
        file_size = int(r.headers["content-length"])
        if file_size < 1000:
            raise ValueError("File doesn't exist? idk")
        chunk_size = 1000
        for chunk in r.iter_content(chunk_size=chunk_size):
            f.write(chunk)
    print(f"Just downloaded {model_type}/model.ckpt.{ext}!", flush=True)
    
# Which news website to 'clone'
DOMAIN_STYLE_TO_COPY = "https://www.hindawi.com/journals/bmri/2013/358945/"
# RSS_FEEDS_OF_REAL_STORIES_TO_EMULATE = [
#   "https://feeds.a.dj.com/rss/RSSWorldNews.xml",
# ]

# Ready to start grabbing RSS feeds
# domain = DOMAIN_STYLE_TO_COPY
# feed_urls = RSS_FEEDS_OF_REAL_STORIES_TO_EMULATE
articles = list()

# Get the read headlines to look more realistic
# for feed_url in feed_urls:
    # articles += get_articles_from_real_blog(domain, feed_url)

f = open("/content/gdrive/MyDrive/grover-fork2/article.json")
article = json.load(f)
f.close()
articles.append(article)

# Toss in the slanderous articles
# articles += get_fake_articles(domain)

# Randomize the order the articles are generated
# random.shuffle(articles)

# Load the pre-trained "huge" Grover model with 1.5 billion params
model_config_fn = '/content/gdrive/MyDrive/grover-fork2/grover/lm/configs/mega.json'
model_ckpt = '/content/gdrive/MyDrive/grover-fork2/grover/models/mega/model.ckpt'
encoder = get_encoder()
news_config = GroverConfig.from_json_file(model_config_fn)

# Set up TensorFlow session to make predictions
tf_config = tf.ConfigProto(allow_soft_placement=True)

with tf.Session(config=tf_config, graph=tf.Graph()) as sess:
    # Create the placehodler TensorFlow input variables needed to feed data to Grover model
    # to make new predictions.
    initial_context = tf.placeholder(tf.int32, [1, None])
    p_for_topp = tf.placeholder(tf.float32, [1])
    eos_token = tf.placeholder(tf.int32, [])
    ignore_ids = tf.placeholder(tf.bool, [news_config.vocab_size])

    # Load the model config to get it set up to match the pre-trained model weights
    tokens, probs = sample(
        news_config=news_config,
        initial_context=initial_context,
        eos_token=eos_token,
        ignore_ids=ignore_ids,
        p_for_topp=p_for_topp,
        do_topk=False
    )

    # Restore the pre-trained Grover 'huge' model weights
    saver = tf.train.Saver()
    saver.restore(sess, model_ckpt)

    # START MAKING SOME FAKE NEWS!!
    # Loop through each headline we scraped from an RSS feed or made up
    for article in articles:
        print(f"Building article from headline '{article['title']}'")

        # # If the headline is one we made up about a specific person, it needs special handling
        # if NAME_TO_SLANDER in article['title']:
        #     # The first generated article may go off on a tangent and not include the target name.
        #     # In that case, re-generate the article until it at least talks about our target person
        #     attempts = 0
        #     while NAME_TO_SLANDER not in article['text']:
        #         # Generate article body given the context of the real blog title
        #         article['text'] = generate_article_attribute(sess, encoder, tokens, probs, article, target="article")

        #         # If the Grover model never manages to generate a good article about the target victim,
        #         # give up after 10 tries so we don't get stuck in an infinite loop
        #         attempts += 1
        #         if attempts > 5:
        #             continue
        # If the headline was scraped from an RSS feed, we can just blindly generate an article
        # else:
            # article['text'] = generate_article_attribute(sess, encoder, tokens, probs, article, target="article")
        article['text'] = generate_article_attribute(sess, encoder, tokens, probs, article, target="article")

        # Now, generate a fake headline that better fits the generated article body
        # This replaces the real headline so none of the original article content remains
        article['title'] = generate_article_attribute(sess, encoder, tokens, probs, article, target="title")

        # Grab generated text results so we can post them to WordPress
        article_title = article['title']
        article_text = article['text']
        article_date = article["iso_date"]
        article_image_url = article["image_url"]
        article_tags = article['tags']

        # Make the article body look more realistic - add spacing, link Twitter handles and hashtags, etc.
        # You could add more advanced pre-processing here if you wanted.
        authors = create_authors()
        university = get_random_school()
        department = get_random_department()
        publish_date = get_date(365 * 2, 365 * 8)
        article_text = generate_html_from_article_text(
            article_title, article_text, authors, publish_date, university, department, article_image_url)

        print(f" - Generated fake article titled '{article_title}'")
        article_title = article_title.replace(" ", "-")
        filename = '/content/gdrive/My Drive/Articles/' + f"{article_title}.html"
        with open(filename, 'w' ) as f:
          f.write(article_text)
        write_to_pdf(filename, "/content/gdrive/My Drive/Articles")

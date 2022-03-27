#!/usr/bin/env python

from faker import Faker
import json, random
from datetime import timedelta
from datetime import datetime

def get_number_authors():
    return random.randint(1, 5)

def get_date(start_days_ago=1, end_days_ago=365):
        days_ago = random.randint(start_days_ago, end_days_ago)
        pub_datetime = datetime.now() - timedelta(days=days_ago)

        publish_date = pub_datetime.strftime('%m-%d-%Y')
        iso_date = pub_datetime.isoformat()
        
        print(publish_date)
        print(iso_date)
        return publish_date

def create_authors(nbr_authors=get_number_authors()):
    authors = list()
    
    for i in range(nbr_authors):
        faker = Faker()
        authors.append(faker.name())            
        # print(f'name: {faker.name()}')
        # address = faker.address().replace("\n", " ")
        # print(f'address: {address}')
    return authors

def get_random_school():
    f = open("working/universities.json")
    universities = json.load(f)
    inx = random.randint(0, len(universities) - 1)
    return universities[inx]["institution"]
        
def get_random_department():
    mf = open("working/majors.json")
    majors = json.load(mf)

    # print(f'school: {data[inx]["institution"]}')
    # print(f'department: {majors["majors"][m_inx]["department"]}')

    # print(f'text: {faker.text()}')                
    # print(f'name: {faker.name()}')
    # address = faker.address().replace("\n", " ")
    # print(f'address: {address}')

    inx = random.randint(0, len(majors["majors"]) - 1)
    return "School of " + majors["majors"][inx]["department"]

# def create_authors(nbr_authors=get_number_authors()):
#     authors = list()
    
#     f = open("working/universities.json")
#     mf = open("working/majors.json")

#     data = json.load(f)
#     majors = json.load(mf)

#     for i in range(nbr_authors):
#         faker = Faker()
            
#         print(f'name: {faker.name()}')
#         address = faker.address().replace("\n", " ")
#         print(f'address: {address}')


#         inx = random.randint(0, len(data) - 1)
#         m_inx = random.randint(0, len(majors["majors"]) - 1)

#         print(f'school: {data[inx]["institution"]}')
#         print(f'department: {majors["majors"][m_inx]["department"]}')

#         # print(f'text: {faker.text()}')

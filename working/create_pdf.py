import os, pdfkit

def write_to_pdf(file_name, out_filepath):
    f = open(file_name)

    options = {
        "enable-local-file-access": ""
    }    
    out_filename = file_name.replace(".heml", ".pdf")
    pdfkit.from_file(file_name, out_filepath + os.sep + out_filename, options=options)

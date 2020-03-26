import PyPDF2
import pandas as pd 
import summerize

def pdf2csv(pdfName):
    
    read_pdf = PyPDF2.PdfFileReader(pdfName)
    content = ""
    for i in range(read_pdf.getNumPages()):
        page = read_pdf.getPage(i)
        page_content = page.extractText()
        content += page_content
    content= content.replace('\n',' ')
    #print(content.split('.'))

    #list_sentences = content.split('.')#sent_tokenize(content)

    #df = pd.DataFrame(list_sentences)


    #df.to_csv("csv\\{0}.csv".format(pdfName))
    print(summerize.summerizze(content,0.9))


import os

arr = os.listdir('pdfs')

for i in range(len(arr)):
    print(arr[i])
    pdf2csv('pdfs\\'+arr[i])
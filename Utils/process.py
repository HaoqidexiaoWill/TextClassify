from pdfminer.converter import TextConverter
from pdfminer.pdfinterp import PDFPageInterpreter
from pdfminer.pdfinterp import PDFResourceManager
from pdfminer.pdfpage import PDFPage
import io
import os
data_dir = '/home/lsy2018/媛媛/data/人工智能教师/'
resource_manager = PDFResourceManager()
fake_file_handle = io.StringIO()
converter = TextConverter(resource_manager,fake_file_handle)
page_interpreter = PDFPageInterpreter(resource_manager,converter)
file_list = os.listdir(data_dir)
for each_file in file_list:
    file_name = os.path.join(data_dir,each_file)
    print(file_name)
    with open(file_name,'rb') as fh:
        for page in PDFPage.get_pages(fh,check_extractable = True):
            page_interpreter.process_page(page)
        text = fake_file_handle.getvalue()
        print(text)
        exit()
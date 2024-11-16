import pickle
import PyPDF2

# Open and read the PDF file
with open('/root/backend/MY23_Prius_OM_Excerpt_for_Driving_Support_Systems_D4_ML_0208.pdf', 'rb') as pdf_file:
    reader = PyPDF2.PdfReader(pdf_file)
    text_data = ""
    for page in reader.pages:
        text_data += page.extract_text()

# Pickle the extracted text
with open('docs.pkl', 'wb') as pkl_file:
    pickle.dump(text_data, pkl_file)
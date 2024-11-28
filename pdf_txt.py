import os
import PyPDF2

def pdfs_a_txt(output_txt_file):
    ruta_carpeta = 'Documentos'
    project_path = os.path.dirname(os.path.abspath(__file__))
    pdf_folder_path = os.path.join(project_path, ruta_carpeta)

    if not os.path.exists(pdf_folder_path):
        print(f"La carpeta '{ruta_carpeta}' no existe en el directorio del proyecto.")
        return

    with open(output_txt_file, 'w', encoding='utf-8') as txt_file:
        for filename in os.listdir(pdf_folder_path):
            if filename.endswith('.pdf'):
                pdf_file_path = os.path.join(pdf_folder_path, filename)
                with open(pdf_file_path, 'rb') as file:
                    reader = PyPDF2.PdfReader(file)
                    for page_num in range(len(reader.pages)):
                        page = reader.pages[page_num]
                        text = page.extract_text()
                        if text:
                            txt_file.write(text)
                            txt_file.write("\n\n")

output_txt_file = 'documento.txt'
pdfs_a_txt(output_txt_file)
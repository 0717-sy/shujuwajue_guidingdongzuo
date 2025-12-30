import os
import pandas as pd
import pdfplumber

def pdf_to_csv(pdf_path, output_path=None):
    if output_path is None:
        output_path = pdf_path.replace('.pdf', '.csv').replace('.PDF', '.csv')

    all_tables = []
        
    with pdfplumber.open(pdf_path) as pdf:
        for page in pdf.pages:
            tables = page.extract_tables()
            if tables:
                for table in tables:
                    df = pd.DataFrame(table)
                    all_tables.append(df)
        
    if not all_tables:
        print(f"未在 {pdf_path} 中找到表格")
        return
        
    all_data = pd.concat(all_tables, ignore_index=True)
        
    all_data.to_csv(output_path, index=False, encoding='utf-8-sig')
    print(f"转换成功: {output_path}")

def batch_convert_pdf_to_csv(folder_path):
    for root, dirs, files in os.walk(folder_path):
        for file in files:
            if file.lower().endswith('.pdf'):
                pdf_path = os.path.join(root, file)
                pdf_to_csv(pdf_path)

if __name__ == '__main__':
    folder_path = r'C:\Users\86151\Desktop\数据挖掘\output\PDF_files'
    batch_convert_pdf_to_csv(folder_path)
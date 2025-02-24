import os

def rename_pdf_files():
    # 获取当前目录下的所有文件
    files = os.listdir('.')
    
    # 遍历文件列表
    for filename in files:
        # 检查文件是否以 .pdf 结尾
        if filename.endswith('.pdf'):
            # 替换文件名中的空格为下划线
            new_filename = filename.replace(' ', '_')
            
            # 如果文件名中确实有空格，进行重命名
            if filename != new_filename:
                os.rename(filename, new_filename)
                print(f"Renamed '{filename}' to '{new_filename}'")

if __name__ == "__main__":
    rename_pdf_files()

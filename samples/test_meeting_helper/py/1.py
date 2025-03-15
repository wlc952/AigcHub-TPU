import gradio as gr
import os
import hashlib
from pathlib import Path

def get_folder_structure(directory):
    """Recursively builds a folder structure string."""
    folder_structure = ""

    for root, dirs, files in os.walk(directory):
        level = root.replace(directory, '').count(os.sep)
        indent = '    ' * level
        folder_structure += f"{indent}{os.path.basename(root)}/\n"
        sub_indent = '    ' * (level + 1)
        for f in files:
            folder_structure += f"{sub_indent}{f}\n"
    return folder_structure

def delete_duplicate_files(directory):
    """Deletes duplicate files based on MD5 hash."""
    if not os.path.isdir(directory):
        return "Invalid directory path."

    hash_dict = {}
    duplicates = []
    for root, _, files in os.walk(directory):
        for file in files:
            filepath = os.path.join(root, file)
            try:
                with open(filepath, 'rb') as f:
                    file_hash = hashlib.md5(f.read()).hexdigest()
                if file_hash in hash_dict:
                    duplicates.append(filepath)
                else:
                    hash_dict[file_hash] = filepath
            except Exception as e:
                return f"Error processing file {filepath}: {str(e)}"

    for dup in duplicates:
        try:
            os.remove(dup)
        except Exception as e:
            return f"Error deleting file {dup}: {str(e)}"

    return get_folder_structure(directory)

def categorize_files(directory):
    """Organizes files into folders based on their file extensions."""
    if not os.path.isdir(directory):
        return "Invalid directory path."

    try:
        for root, _, files in os.walk(directory):
            for file in files:
                filepath = os.path.join(root, file)
                if os.path.isfile(filepath):
                    ext = Path(file).suffix[1:]  # Get extension without dot
                    if ext == "":
                        ext = "no_extension"
                    target_dir = os.path.join(directory, ext)
                    os.makedirs(target_dir, exist_ok=True)
                    target_path = os.path.join(target_dir, file)
                    os.rename(filepath, target_path)
    except Exception as e:
        return f"Error categorizing files: {str(e)}"

    return get_folder_structure(directory)

# Define Gradio Interface
with gr.Blocks() as demo:
    gr.Markdown("# 文件管理工具")
    with gr.Tab("重复文件删除"):
        with gr.Row():
            folder_input_dup = gr.Textbox(label="文件夹路径", placeholder="输入要去重的文件夹路径")
        delete_button = gr.Button("删除重复文件")
        dup_output = gr.Textbox(label="文件夹结构", lines=20)

        delete_button.click(delete_duplicate_files, inputs=folder_input_dup, outputs=dup_output)

    with gr.Tab("文档归类"):
        with gr.Row():
            folder_input_cat = gr.Textbox(label="文件夹路径", placeholder="输入要归类的文件夹路径")
        categorize_button = gr.Button("归类文件")
        cat_output = gr.Textbox(label="文件夹结构", lines=20)

        categorize_button.click(categorize_files, inputs=folder_input_cat, outputs=cat_output)

    gr.Markdown("© 2023 文件管理工具")

demo.launch()

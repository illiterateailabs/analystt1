import os
import re
import glob

TOOLS_BASE_DIR = "backend/agents/tools" # Relative to project root
MODIFIED_FILES_COUNT = 0

# --- Regex Patterns ---
# For 'class XYZTool(BaseTool):'
CLASS_INHERITANCE_RE = re.compile(r"class\s+(\w+)\s*\(\s*BaseTool\s*\):")
# For 'args_schema: type[BaseModel] = XYZInput' or 'args_schema = XYZInput'
ARGS_SCHEMA_RE = re.compile(r"^\s*args_schema\s*[:=]\s*.*$")
# For 'super().__init__(...)'
SUPER_INIT_RE = re.compile(r"^\s*super\(\s*\)\s*\.__init__\s*\(.*\)$")
# For 'def _run('
DEF_RUN_RE = re.compile(r"def\s+_run\s*\(")


def remove_basetool_from_import_line(line_content: str) -> Optional[str]:
    """
    Processes a line to remove BaseTool (and its aliases) if imported from crewai_tools.
    Returns the modified line, or None if the line should be deleted entirely.
    """
    # Match lines like 'from crewai_tools import ...'
    match_from_crewai_tools = re.match(r"^(\s*from\s+crewai_tools\s+import\s+)(.*)$", line_content)
    if not match_from_crewai_tools:
        return line_content  # Not an import from crewai_tools

    indentation_and_prefix = match_from_crewai_tools.group(1)  # e.g., "    from crewai_tools import "
    imports_and_comment_part = match_from_crewai_tools.group(2)

    # Separate the actual import items from any trailing comments
    comment_start_index = imports_and_comment_part.find('#')
    if comment_start_index != -1:
        imports_str = imports_and_comment_part[:comment_start_index].strip()
        comment_str = " " + imports_and_comment_part[comment_start_index:] # Preserve space before #
    else:
        imports_str = imports_and_comment_part.strip()
        comment_str = ""
    
    if not imports_str: # Line was like "from crewai_tools import # comment"
        return line_content


    # Split imported items: "BaseTool", "BaseTool as BT", "OtherTool"
    # This needs to handle parentheses for multi-line imports if they exist, though less common for tools.
    # Assuming simple single-line imports for now.
    # A more robust parser would be needed for complex import statements (e.g. with parentheses)
    # but for typical tool imports, splitting by comma should be okay.
    
    original_items = [item.strip() for item in imports_str.split(',') if item.strip()]
    
    items_to_keep = []
    basetool_was_imported = False

    for item in original_items:
        # Check for "BaseTool" or "BaseTool as Alias"
        if item == "BaseTool" or re.match(r"^BaseTool\s+as\s+\w+$", item):
            basetool_was_imported = True
        else:
            items_to_keep.append(item)

    if not basetool_was_imported:
        return line_content # BaseTool wasn't on this line, no change needed.

    if not items_to_keep:
        # BaseTool (or its alias) was the only item imported.
        # If there's a comment, keep the comment on its own line, otherwise delete.
        if comment_str.strip(): # If comment_str is not just whitespace
             # Return only the initial indentation + comment
            return line_content[:line_content.find("from")] + comment_str.rstrip() + "\n"
        return None  # Signal to delete the line
    else:
        # Reconstruct the import line with remaining items
        return indentation_and_prefix + ", ".join(items_to_keep) + comment_str.rstrip() + "\n"


def process_python_file(filepath: str):
    global MODIFIED_FILES_COUNT
    
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            lines = f.readlines()
    except Exception as e:
        print(f"Error reading file {filepath}: {e}")
        return

    new_content_lines = []
    file_modified_this_run = False
    in_init_method = False
    current_method_indent = -1

    for line_no, line_content in enumerate(lines):
        original_line = line_content

        # Transformation 1: Remove BaseTool import from crewai_tools
        processed_import_line = remove_basetool_from_import_line(line_content)
        if processed_import_line is None: # Line to be deleted
            file_modified_this_run = True
            # print(f"DEBUG: Line {line_no+1} deleted (BaseTool import): {original_line.strip()}")
            continue 
        elif processed_import_line != original_line:
            line_content = processed_import_line
            file_modified_this_run = True
            # print(f"DEBUG: Line {line_no+1} modified (BaseTool import removed): {line_content.strip()}")

        # Transformation 2: Change class inheritance
        # Example: class MyTool(BaseTool): -> class MyTool:
        line_content, count = CLASS_INHERITANCE_RE.subn(r"class \1:", line_content)
        if count > 0:
            file_modified_this_run = True
            # print(f"DEBUG: Line {line_no+1} modified (inheritance): {line_content.strip()}")

        # Transformation 3: Remove args_schema attribute lines
        if ARGS_SCHEMA_RE.match(line_content):
            file_modified_this_run = True
            # print(f"DEBUG: Line {line_no+1} deleted (args_schema): {original_line.strip()}")
            continue

        # Track __init__ method scope for super().__init__() removal
        line_lstrip = line_content.lstrip()
        current_indent = len(line_content) - len(line_lstrip)

        if line_lstrip.startswith("def __init__("):
            in_init_method = True
            current_method_indent = current_indent
        elif in_init_method and line_lstrip != "" and current_indent <= current_method_indent:
            # Exited the __init__ method (dedent or new def at same level)
            in_init_method = False
            current_method_indent = -1
        
        # Transformation 4: Remove super().__init__() calls
        if in_init_method and SUPER_INIT_RE.match(line_lstrip):
            file_modified_this_run = True
            # print(f"DEBUG: Line {line_no+1} deleted (super init): {original_line.strip()}")
            continue
            
        # Transformation 5: Rename _run method to run
        line_content, count = DEF_RUN_RE.subn(r"def run(", line_content)
        if count > 0:
            file_modified_this_run = True
            # print(f"DEBUG: Line {line_no+1} modified (_run to run): {line_content.strip()}")
            
        new_content_lines.append(line_content)

    if file_modified_this_run:
        try:
            with open(filepath, 'w', encoding='utf-8') as f:
                f.writelines(new_content_lines)
            print(f"Modified: {filepath}")
            MODIFIED_FILES_COUNT += 1
        except Exception as e:
            print(f"Error writing modified file {filepath}: {e}")


def main():
    # Determine project root: assumes script is in project_root/scripts/ or project_root/
    script_path = os.path.abspath(__file__)
    project_root = os.path.dirname(os.path.dirname(script_path)) # if in scripts/
    if not os.path.basename(project_root) == "analyst-agent-illiterateai": # A simple check
         project_root = os.path.dirname(script_path) # if script is at root
         if not os.path.basename(project_root) == "analyst-agent-illiterateai" and os.path.exists(os.path.join(project_root, TOOLS_BASE_DIR)):
             pass # Looks like project_root is correct
         elif os.path.exists(os.path.join(os.getcwd(), TOOLS_BASE_DIR)):
             project_root = os.getcwd() # Fallback to CWD if structure is unexpected
         else:
            print(f"Warning: Could not reliably determine project root. Assuming script is run from project root or 'scripts/' subdir.")
            print(f"Current project_root guess: {project_root}")
            # If TOOLS_BASE_DIR is not found relative to project_root, try CWD
            if not os.path.isdir(os.path.join(project_root, TOOLS_BASE_DIR)):
                project_root = os.getcwd()


    tools_dir_abs_path = os.path.join(project_root, TOOLS_BASE_DIR)

    if not os.path.isdir(tools_dir_abs_path):
        print(f"Error: Tools directory not found at '{tools_dir_abs_path}'.")
        print(f"Please run this script from the project root directory or its 'scripts' subdirectory.")
        return

    print(f"Scanning for Python files in: {tools_dir_abs_path}\n")
    
    # Find all .py files recursively in the tools directory
    for dirpath, _, filenames in os.walk(tools_dir_abs_path):
        for filename in filenames:
            if filename.endswith(".py"):
                filepath = os.path.join(dirpath, filename)
                process_python_file(filepath)

    print(f"\n--- Summary ---")
    if MODIFIED_FILES_COUNT > 0:
        print(f"Script complete. Modified {MODIFIED_FILES_COUNT} file(s).")
        print("Please review the changes and test thoroughly.")
    else:
        print("No files required modification based on the specified criteria.")

if __name__ == "__main__":
    main()

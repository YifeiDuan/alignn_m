import os

def ensure_data_prefix(content, filename):
    """
    Ensure the CIF content starts with a 'data_' line.
    If not, add one using the filename (without extension).
    """
    lines = [l.strip() for l in content.strip().splitlines()]
    if not lines:
        return content  # skip empty files

    if not lines[0].startswith("data_"):
        name = os.path.splitext(os.path.basename(filename))[0]
        return f"data_{name}\n" + "\n".join(lines) + "\n"
    else:
        return content if content.endswith("\n") else content + "\n"


def fix_cif_directory(input_dir, output_dir):
    """
    Add 'data_' header to all CIF files missing it, and save to output_dir.
    """
    os.makedirs(output_dir, exist_ok=True)
    for fname in os.listdir(input_dir):
        if not fname.lower().endswith(".cif"):
            continue

        input_path = os.path.join(input_dir, fname)
        output_path = os.path.join(output_dir, fname)

        try:
            with open(input_path, "r") as f:
                content = f.read()
            fixed_content = ensure_data_prefix(content, fname)
            with open(output_path, "w") as f:
                f.write(fixed_content)
            print(f"✅ Processed: {fname}")
        except Exception as e:
            print(f"⚠️ Failed on {fname}: {e}")
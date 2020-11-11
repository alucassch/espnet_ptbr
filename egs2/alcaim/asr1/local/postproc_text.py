import os
import sys
import shutil
from tempfile import mkstemp

def process_text(text: str) -> str:
    return text.lower().replace(',','').replace('.','').replace('-','').replace('?','').replace('!','').replace(':','').replace('"','')

def main(text_file: str) -> None:
    if not os.path.exists(text_file):
        print(f"File {text_file} does not exist")
        sys.exit(1)

    try:
        fid, fname = mkstemp(dir='/tmp')
        os.close(fid)

        with open(fname,'w') as fout:
            with open(text_file) as f:
                for line in f:
                    uttid = line.strip().split()[0]
                    text = ' '.join(line.strip().split()[1:]).strip()
                    text = process_text(text)
                    fout.write(f"{uttid} {text}\n")
    finally:
        shutil.move(fname, text_file)


if __name__ == '__main__': 
    if len(sys.argv) != 2:
        print("Usage {__name__} text_file")
    main(sys.argv[1])
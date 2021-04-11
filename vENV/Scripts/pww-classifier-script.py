#!C:\Users\Giovanni\Desktop\muse2_fft_experiments-master\vENV\Scripts\python.exe
# EASY-INSTALL-ENTRY-SCRIPT: 'python-weka-wrapper3==0.1.15','console_scripts','pww-classifier'
__requires__ = 'python-weka-wrapper3==0.1.15'
import re
import sys
from pkg_resources import load_entry_point

if __name__ == '__main__':
    sys.argv[0] = re.sub(r'(-script\.pyw?|\.exe)?$', '', sys.argv[0])
    sys.exit(
        load_entry_point('python-weka-wrapper3==0.1.15', 'console_scripts', 'pww-classifier')()
    )

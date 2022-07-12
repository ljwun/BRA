pip3 install torch torchvision torchaudio --extra-index-url https://download.pytorch.org/whl/cu113
pip3 install loguru
pip3 install thop
pip3 install opencv-python==4.5.5.64
pip3 install tabulate
pip3 install pycocotools
pip3 install scipy
pip3 install -U wheel 
pip3 install lap
pip3 install Cython
curl -L -O https://files.pythonhosted.org/packages/fa/b9/fc7d60e8c3b29cc0ff24a3bb3c4b7457e10b7610fbb2893741b623487b34/cython_bbox-0.1.3.tar.gz
tar -xz -f .\cython_bbox-0.1.3.tar.gz
rm .\cython_bbox-0.1.3.tar.gz
cd .\cython_bbox-0.1.3
(Get-Content '.\setup.py' -Raw) -replace 'extra_compile_args=\[[^\]]*]', "extra_compile_args = {'gcc': ['/Qstd=c99']}" | Out-File '.\setup.py'
python .\setup.py build_ext install
cd ..
rm -r .\cython_bbox-0.1.3
pip3 install argparse
# curl -L -O http://ciscobinary.openh264.org/openh264-1.8.0-win64.dll.bz2
# python -c "import bz2,shutil,os
# fr=bz2.BZ2File('openh264-1.8.0-win64.dll.bz2')
# fw=open(os.path.join('openh264-1.8.0-win64.dll'),'wb')
# shutil.copyfileobj(fr,fw)"
# rm .\openh264-1.8.0-win64.dll.bz2
pip3 install tensorboard
pip3 install tqdm
pip3 install scikit-image
pip3 install pyyaml
pip3 install pandas
pip3 install ujson

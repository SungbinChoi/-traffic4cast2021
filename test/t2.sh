cd t2m1
python run1.py
python run2.py
cd ../t2m2
python run1.py
python run2.py
cd ../t2m3
python run1.py
python run2.py
cd ../t2m4
python run1.py
python run2.py
cd ../combine
python t2.py
cd ../test_runs/t2
zip -r submit2.zip .

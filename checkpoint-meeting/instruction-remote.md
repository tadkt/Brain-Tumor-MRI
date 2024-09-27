--Start of command--
Prerun: Install connector.bat

Step 1: Run connector.bat then choose 2

Step 2: When it connected to frontend, Enter:
ssh -L localhost:29998:localhost:29998 ict6 

Step 3: Enable persistence screening: Enter: screen

Step 4: Turn on Jupyter Lab. Enter:
jupyter lab --port 29998

Step 5: Find the url that has "http://localhost:29998/" in the log when you ran Jupyter Lab. Copy it and access it on Browser

--End of command--

# MSCI-436-House-Price-ML-Project
This repository holds our linear regression ML model python program as well as the basic application script we have using Streamlit. It is a basic Decision Support Tool to help with understanding house pricing based on multiple factors.


## Streamlit

This was written with Python 3.9-3.11.

If you want, make a venv and enter into it. Then:
1. Install requirements.txt
    ```shell
    pip install -r requirements.txt
    ```
1. Make dummy model 
    ```shell
    python .\make_dummy_model.py
   ```
   
2. Run Streamlit 
   ```shell
    streamlit run dash.py
    ```
3. Freeze dependencies
    ```shell
    pip freeze > requirements.txt
    ```
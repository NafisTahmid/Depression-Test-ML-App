# Core Packages
import streamlit as st
import streamlit.components.v1 as stc

# Import Our Mini Apps
from eda_app import run_eda_app
from ml_app import run_ml_app


html_temp = """
           <div style="background-color:#3872fb;padding:10px;border-radius:10px">
		        <h1 style="color:white;text-align:center;">Predictive Analysis on Depression App</h1>
		        <h4 style="color:white;text-align:center;">Among University Students</h4>
		   </div>
            """

desc_temp = """
                ### Predictive Analysis on Depression App Among University Students
                This dataset contains the sign and symptoms data of newly depressed or would be depressed patient who are in university years.
                #### App Content
                    - EDA Section: Exploratory Data Analysis of Data
                    - ML Section: ML Predictor App
            """

def main():
    # st.title("Main App")
    stc.html(html_temp)
    menu = ["Home", "EDA", "ML", "About"]
    choice = st.sidebar.selectbox("Menu", menu)

    if choice == "Home":
        st.subheader("Home")
        # st.write(desc_temp)
        st.markdown(desc_temp, unsafe_allow_html=True)
    elif choice == "EDA":

        run_eda_app()
    elif choice == "ML":
        run_ml_app()
    else:
        st.subheader("About")
        st.markdown("#### Supervisor")
        st.text("MD. Shahriar Rahman Rana")
        st.markdown("#### Co Supervisor")
        st.text("Rafeed Rahman")
        st.markdown("#### Students")
        st.text("Syed Aref Ahmed - 19201124 - syed.aref.ahmed@g.bracu.ac.bd ")
        st.text("Khondokar Jamal E Mustafa - 19241008 - khondokar.jamal.e.mustafa@g.bracu.ac.bd")
        st.text("MD. Nafis Tahmid - 19301053 - md.nafis.tahmid@g.bracu.ac.bd")
        st.text(" Ibtesum Arif - 19201054 - ibtesum.arif@g.bracu.ac.bd")
        


if __name__ == "__main__":
    main()
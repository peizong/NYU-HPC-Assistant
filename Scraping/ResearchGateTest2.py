import os
from bs4 import BeautifulSoup
from sympy import sympify
import undetected_chromedriver as uc
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
import time

def extract_topic_name(url):
    return url.split("/")[-1].replace("-", " ")

def get_topic_links():
    return "https://www.researchgate.net/topic/Journalism"

    #Enable the code below once the rest of the code is complete. Remember to include a significant delay between pages (maybe random 0-30 seconds)
    base_url = "https://www.researchgate.net/topic/"
    topics = [
        "Journalism", "MATLAB", "Publisher", "Filing", "Writing", "Abaqus", "Reasoning",
        "Publication", "Artificial-Intelligence", "Machine-Learning", "Students", "ANSYS",
        "Research-Papers", "Cell-Culture", "Attitude", "Primer", "Plasmids",
        "Molecular-Biological-Techniques", "Mice", "Cell-Line", "India", "Staining",
        "Statistics", "Websites", "Methods", "Virus", "Indexes", "Interpretation",
        "Graphs", "RNA", "Psychology", "Molecular-Dynamics-Simulation", "Assays",
        "Universities", "Polymerization", "Gels", "Sample-Size", "Concrete", "Publishing",
        "FLUENT", "Column", "X-ray-Diffraction", "Stress", "Modeling", "Data-Analysis",
        "Comsol-Multiphysics", "Chemistry", "Brain", "Extraction", "Microbiology",
        "Training", "Bioinformatics", "Cell-Culture-Techniques", "Education", "Manuscripts",
        "Regression", "High-Performance-Liquid-Chromatography", "Earth", "Deep-Learning",
        "Particle", "Agriculture", "Vaccines", "Articles", "Business", "Transfection",
        "ELISA", "Biochemistry", "Atoms", "Environmentalism", "Philosophy", "Coronavirus",
        "Bioinformatics-and-Computational-Biology", "Fixatives", "English", "Comment",
        "Heat", "Ethics", "Tissue", "Vectorization", "Acids", "Calculations", "Digital",
        "Neural-Networks", "Flow-Cytometry", "Crop", "Gaussian", "Mass", "Fees-and-Charges",
        "Cell-Biology", "Image-Processing", "Membranes", "Liquids", "Organic-Chemistry",
        "Devices", "Sensors", "Computer-Science", "Gas", "3D", "Research", "Neuroscience",
        "Computer", "Happiness", "Solar", "Cloning", "Advanced-Statistical-Analysis",
        "Microorganisms", "Air", "Medicine", "R", "Filtering", "Antennas", "Economy",
        "Social-Media", "Ecology", "Glass", "Real-Time-PCR", "Graphite", "Zero",
        "Time-Series", "Ligand", "Computational-Fluid-Dynamics", "Renewable-Energy",
        "DMSO", "Maps", "Scanning-Electron-Microscopy", "Adsorption", "Data-Mining",
        "Packaging", "Soil-Analysis", "Contamination", "Reagents", "Spectrum",
        "Electrochemistry", "Semiconductor", "Hardness", "Finite-Element-Method",
        "Ions", "Iron", "Materials", "Density-Functional-Theory", "COVID-19", "Thinking",
        "Papers", "Books", "Running", "Learning", "PCR", "Images", "Molecular-Biology",
        "Extracts", "SPSS", "Teaching", "Dataset", "Nanoparticles", "Solvents",
        "Correction", "Western-Blot", "Antibodies", "Software", "Mathematics", "Buffer",
        "Publications", "Free-Will", "Pandemics", "Climate-Change", "Citations", "Python",
        "Drugs", "Collaboration", "Plating", "Names", "Simulators", "Conferences",
        "Survey", "pH", "Connectivity", "Journal-Impact-Factor", "Science", "Electrodes",
        "Academic-Journals", "Light", "Thin-Films", "Carbon", "Matrix", "Thesis-Research",
        "Polymerase-Chain-Reaction", "Color", "Graphene", "Biotechnology",
        "Statistical-Analysis", "Pressure", "Coating", "Program", "School",
        "clinical-coding", "Powders", "Methodology", "Physics", "Materials-Science",
        "Scientific-Research", "Steel", "Research-Topics", "Gromacs", "Reliability",
        "Mechanical-Engineering", "Structural-Equation-Modeling", "DNA", "Scopus",
        "Questionnaire", "Nano", "Children", "Fluorescence", "Confusion", "Classification",
        "Engineering", "ANOVA", "Personal-Autonomy", "Bacteria", "Resistance", "Instruments",
        "Oil", "Data", "Analytical-Chemistry", "Emotion", "Dissertations", "Enzymes",
        "Gene-Expression", "Organic", "Sectioning", "Electrical-Engineering", "Solar-Cells",
        "Fungi", "Face", "Coding", "Mind", "Industry", "Proteins", "Fish",
        "Microsoft-Office-Excel", "Plant-Extracts", "Remote-Sensing", "Solubility", "PEAKS",
        "Finite-Element-Analysis", "Salts", "Dyes", "Waves", "SDS-PAGE", "Meta-Analysis",
        "Ethanol", "microRNA", "Biology", "Carbon-Dioxide", "Economics", "Dosing",
        "Peptides", "Immunology", "Electron", "Seeds", "Battery", "Nanotechnology",
        "Internet", "Machines", "Plants", "Innovation", "Transformers",
        "Materials-Engineering", "Absorption", "Bonds", "Social-Science",
        "Internet-of-Things", "Precipitation", "Soil", "Green-IT", "Laboratory",
        "Big-Data", "Drying", "Docking", "Higher-Education", "Qualitative-Research"
    ]
    return [base_url + topic for topic in topics]

def scrape_topic(topic_url):
    all_qa_data = []
    page = 1
    topic_name = extract_topic_name(topic_url)
    
    options = uc.ChromeOptions()
    options.add_argument("--no-sandbox")
    options.add_argument("--disable-dev-shm-usage")
    options.add_argument("user-agent=Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                         "AppleWebKit/537.36 (KHTML, like Gecko) "
                         "Chrome/122.0.0.0 Safari/537.36")
    
    try:
        driver = uc.Chrome(options=options, use_subprocess=True)
        time.sleep(5)
    except Exception as e:
        print(f"Error initializing Chrome driver: {e}")
        print("UPDATE YOUR CHROME BROWSER")


    while True:
        url = f"{topic_url}/{page}" if page > 1 else topic_url
        driver.get(url)
        
        try:
            WebDriverWait(driver, 15).until(
                EC.presence_of_element_located((By.CSS_SELECTOR, 'main'))
            )
            driver.execute_script("window.scrollTo(0, document.body.scrollHeight);")
            time.sleep(2)
        except Exception as e:
            print(f"Error loading page: {e}")
            break

        html = driver.page_source
        save_html_to_file(html, topic_name, page)
        
        soup = sympify(html)
        
        qa_data = extract_qa_data(soup)
        if not qa_data:
            break
        
        all_qa_data.extend(qa_data)
        page += 1
    
    driver.quit()
    
    return {
        "topic": topic_name,
        "qa_data": all_qa_data
    }


def extract_qa_data(soup: BeautifulSoup):

    #THIS EXTRACTION PROCESS IS BROKEN, REPLACE WITH PROPER EXTRACTION PROCESS

    qa_elements = soup.select('.nova-c-card.nova-c-card--spacing-xl.nova-c-card--elevation-1-above')
    qa_data = []

    for qa in qa_elements:
        question_element = qa.select_one('.nova-legacy-v-question-item__title')
        question = question_element.text.strip() if question_element else "N/A"

        answer_element = qa.select_one('.nova-legacy-v-question-item__answer-preview')
        answer = answer_element.text.strip() if answer_element else "N/A"

        user_element = qa.select_one('.nova-legacy-e-link.nova-legacy-e-link--color-inherit.nova-legacy-e-link--theme-bare')
        user = user_element.text.strip() if user_element else "N/A"

        user_image_element = qa.select_one('img.nova-legacy-e-avatar__img')
        user_image_url = user_image_element['src'] if user_image_element and 'src' in user_image_element.attrs else "N/A"

        stats_element = qa.select_one('.nova-legacy-l-flex.nova-legacy-l-flex--gutter-xs')
        stats = stats_element.text.strip() if stats_element else "N/A"

        qa_data.append({
            "question": question,
            "answer_preview": answer,
            "user": user,
            "user_image_url": user_image_url,
            "stats": stats
        })

    return qa_data

def save_html_to_file(html, topic_name, page):
    base_dir = "researchgate_topics_html"
    topic_dir = os.path.join(base_dir, topic_name.replace(' ', '_'))
    os.makedirs(topic_dir, exist_ok=True)
    
    filename = f"page_{page}.html"
    filepath = os.path.join(topic_dir, filename)
    
    soup = BeautifulSoup(html, 'html.parser')
    
    main_content = soup.select_one('main') or soup.select_one('body')
    
    if main_content:
        clean_html = f"""
        <html>
        <body>
            {main_content.prettify()}
        </body>
        </html>
        """
    else:
        clean_html = f"<html><body><p>Main content not found</p><p>Full HTML:</p>{soup.prettify()}</body></html>"

    with open(filepath, 'w', encoding='utf-8') as f:
        f.write(clean_html)
    


def main():
    topic_links = get_topic_links()
    for topic_url in topic_links:
        results = scrape_topic(topic_url)
    time.sleep(1) 

if __name__ == "__main__":
    main()

from flask import Flask, render_template, request
#import fonctions as mod
from prepros import nettoyage_text
from test import most_com, tags_lda


# 1- Netoyer text
# 2- faire passer le text dans le best truc
# 3- recuperer les tags

app = Flask(__name__)

@app.route('/')
def home():
    return render_template('home.html')

@app.route('/shortenurl', methods=['GET', 'POST'])
def shortenurl():
    if request.method == 'POST':
        r=request.form
        
        print("titre de la question : ", r['title'])
        print("développement de la question : ", r['question'])
        
        liste = [r['title'] ,r['question'], r['title'],r['title']]
        question = " ".join(liste)
        question = nettoyage_text(question)
        print(question)

        phrase1 = most_com(question)
        phrase2 = tags_lda(question)
        print(phrase2)

        return render_template('shortenurl.html', shortcode1=phrase1, shortcode2=phrase2 )

    elif request.method == 'GET':
        return 'A GET request was made'
    
    else:
        return 'Not a valid request method for this route' 

if __name__ == "__main__":
    app.run(debug=True)
 

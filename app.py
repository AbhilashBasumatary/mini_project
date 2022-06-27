from flask import Flask,request,make_response,jsonify
from werkzeug.utils import secure_filename
import prediction

app = Flask(__name__)

ALLOWED_EXTENSIONS = {'wav','x-wav'}

@app.route('/')
def hello():
    return 'Hello World!'

def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/submit', methods=['POST'])
def submit():
    if request.method == 'POST':
            f = request.files['file']
            if f.filename == "":
                d="unSuccessfully submitted no file name"
                return make_response(jsonify(status=False,
        mesg="Successfull",), 400)
        #     if not allowed_file(f.filename):
        #         d="unSuccessfully submitted not allowed"
        #         print(f.filename)
        #         return make_response(jsonify(status=False,
        # mesg="Un   successfull"+str(f.filename)), 400)
            else:
                full_filename = secure_filename(f.filename)
                name = prediction.make_predictions(f)[0]
                d="Successfully submitted "+str(name)
                return make_response(jsonify(status=True,
        mesg="Successfull",
        data=name), 200)

                
if __name__ == '_main_':
    app.run(debug=True)





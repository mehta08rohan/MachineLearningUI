from flask import Flask , request,render_template,flash,redirect,url_for
from werkzeug.utils import secure_filename
import model as md
import datetime
import os
import pandas as pd
import shutil

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'file_uploaded/'
app.secret_key = "secret key"
global trainset
global testset
@app.route('/',methods=['GET','POST'])
def HomePage():
	print('rohan')
	if request.method == "POST":
		# for f in request.files['file']:
		if 'files[]' not in request.files:
			flash('File Missing!')
			

		files = request.files.getlist('files[]')
		for f in files:
			
			file_name = f.filename.rsplit('.')
			date = str(datetime.datetime.now())
			date = date.replace('-', '').replace(':', '').replace('.', '').replace(' ','_')
			file_name = file_name[0] + date + '.' + file_name[1]
			f.save(os.path.join(app.config['UPLOAD_FOLDER'], secure_filename(file_name)))
			
			if files.index(f)==0:
				trainset = os.path.join(app.config['UPLOAD_FOLDER'], secure_filename(file_name))
			else:
				testset = os.path.join(app.config['UPLOAD_FOLDER'], secure_filename(file_name))

		clf_dec,clfKNN,city_list , store_location_list , location_employee_code_list , credit_score_list, credit_score_range_list,city_labels,store_location_labels,location_employee_code_labels,credit_score_labels,credit_score_range_labels = md.train(trainset)
		tree, knn = md.predict_test(testset,clf_dec,clfKNN,city_list , store_location_list , location_employee_code_list , credit_score_list, credit_score_range_list,city_labels,store_location_labels,location_employee_code_labels,credit_score_labels,credit_score_range_labels)


		df = pd.read_csv('solution.csv')
		df.to_html('prediction_page.html')
		shutil.move('prediction_page.html', 'templates/prediction_page.html')

		return render_template('prediction_page.html')

		# prediction_tree, prediction_knn = md.perdict_value()
		# return render_template('prediction_page.html',tree=prediction_tree,knn=prediction_knn)

	return render_template('home_page.html')

if __name__ =="__main__":
	app.run(debug=True)


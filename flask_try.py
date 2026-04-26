from flask import Flask, render_template, request, jsonify

app = Flask(__name__)

@app.route('/')
def index():
    return render_template('new.html')

@app.route('/submit', methods=['POST'])
def submit():
    data = request.form
    # 处理表单数据
    return jsonify({'message': '数据已提交'})

if __name__ == '__main__':
    app.run(host='127.0.0.1', port='23333', debug=True, threaded=True, use_reloader=False)
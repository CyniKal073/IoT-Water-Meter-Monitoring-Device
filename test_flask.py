from flask import Flask
from flask import render_template
from flask import Response
from flask import jsonify
import cv2
from PIL import Image
import socket
import numpy as np

state_list = ['自动关盖', '自动开盖', '人为开盖', '人为关盖']
global w_data
global t_data
global h_data
w_data = '0'
t_data = '0'
h_data = '0'
'''
def UDP_Recv():
    udp_socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)  # 确定通信协议类型
    recv_addr = ('127.0.0.1', 8081)
    while True:
        try:
            udp_socket.bind(recv_addr)
        except OSError:
            pass
        else:
            break
    udp_socket.setblocking(0)  # 设置为非阻塞模式
    recv_byte, send_addr = udp_socket.recvfrom(921600)
    receive_data = np.frombuffer(recv_byte, dtype='uint8')
    r_img = cv2.imdecode(receive_data, 1)
    image = cv2.imencode('.jpg', r_img)[1].tobytes()
    udp_socket.close()
    return image
'''

udp_socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)  # 确定通信协议类型
recv_addr = ('192.168.137.1', 8081)
while True:
    try:
        udp_socket.bind(recv_addr)
    except OSError:
        pass
    else:
        break
udp_socket.setblocking(0)  # 设置为非阻塞模式

app = Flask(__name__)

def generate():
    global w_data
    global t_data
    global h_data
    image = cv2.imread('pic.png')
    image = cv2.imencode('.jpg', image)[1].tobytes()
    while True:
        try:
            recv_byte, send_addr = udp_socket.recvfrom(921600)
            receive_data = np.frombuffer(recv_byte, dtype='uint8')
            r_img = cv2.imdecode(receive_data, 1)
            image = cv2.imencode('.jpg', r_img)[1].tobytes()
        except BlockingIOError:
            pass
        except cv2.error:
            data = recv_byte.decode()
            for i in range(len(data)):
                if data[i] == 'w':
                    j = i
                    while data[j] != 't':
                        j = j + 1
                    w_data = data[i+1:j]
                if data[i] == 't':
                    j = i
                    while data[j] != 'h':
                        j = j + 1
                    t_data = data[i+1:j]
                if data[i] == 'h':
                    h_data = data[i+1]

        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + image + b'\r\n\r\n')


@app.route('/')
def main():
    return render_template('test.html', test='hello', water_data=w_data,tem_data=t_data, state_data=state_list[int(h_data)])
'''
@app.route('/water_data')
def water_data():
    def w_generate():
        old = '0'
        while True:
            if w_data[:4] == 'data':
                if w_data != old:
                    old = w_data
                    yield f'{w_data[4:]}\n'
    return Response(w_generate())
    '''

'''
@app.route('/update_data', methods=['POST'])
def update_data():
    print(w_data)
    return jsonify({'data': w_data})
'''

@app.route('/img_data')
def img_data():
    return Response(generate(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == '__main__':
    app.run(host='192.168.137.1', port='23333', debug=True, threaded=True, use_reloader=False)

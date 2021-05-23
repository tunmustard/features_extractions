#!/usr/bin/env python
from importlib import import_module
import os, time
from flask import Flask, render_template, Response,flash, redirect,session, request, url_for


# import camera driver
from camera_recognition import Camera_compare as Camera


# Raspberry Pi camera module (requires picamera package)
# from camera_pi import Camera

app = Flask(__name__)

# Set the secret key to some random bytes. Keep this really secret!
app.secret_key = b'_5#sdaszx335\n\xec]/'


@app.route('/')
def index():
    """Video streaming home page."""
    return render_template('index.html')

@app.route('/login', methods=['GET', 'POST'])
def login():
    error = None
    if request.method == 'POST':
        if request.form['username'] != 'admin' or \
                request.form['password'] != 'secret':
            error = 'Invalid credentials'
        else:
            session['username'] = request.form['username']
            flash('You were successfully logged in')
            return redirect(url_for('index'))
    return render_template('login.html', error=error)

@app.route('/logout')
def logout():
    # remove the username from the session if it's there
    session.pop('username', None)
    return redirect(url_for('index'))

@app.route('/report')
def report():
    """Report homepage"""
    encodings_core_dict = {} #
    return render_template('report.html',encodings_core_dict=encodings_core_dict)

def gen(camera):
    """Video streaming generator function."""
    while True:
        #time.sleep(0.5)
        frame = camera.get_frame()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')


@app.route('/video_feed')
def video_feed():
    """Video streaming route. Put this in the src attribute of an img tag."""
    return Response(gen(Camera()),
                    mimetype='multipart/x-mixed-replace; boundary=frame')


if __name__ == '__main__':
    app.run(host='192.168.178.80', threaded=True)

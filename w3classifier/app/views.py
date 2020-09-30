from flask import send_from_directory, send_file, jsonify, Response

from . import app, dbs


# Path for our main Svelte page
@app.route("/")
def base():
    print(dbs.path_to_data)
    return jsonify(dbs.path_to_data)


# Path for all the static files (compiled JS/CSS, etc.)
@app.route("/<path:path>")
def home(path):
    return send_from_directory('client/public', path)


@app.route("/store_label/<label>")
def store_label(label):
    return jsonify({'res': dbs.store_label(label)})


@app.route('/image/<nm>')
def get_image_by_id(nm):
    if nm == 'undefined':
        return ''
    return send_file(dbs.image(nm), mimetype='image/gif')


@app.route('/marked_image/<nm>')
def marked_image(nm):
    if nm == 'undefined':
        return ''
    jpeg = dbs.marked_image(nm)

    resp = (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + jpeg + b'\r\n\r\n')

    return Response(resp, mimetype='multipart/x-mixed-replace; boundary=frame')



@app.route('/set_value/<im>/<label>/<code>')
def set_value(im, label, code):
    v = dbs.set_value(im, label, code)
    print(f'set value res = {v}')
    return jsonify(v)


@app.route('/set_filter/<label>/<value>/<seek_label>/<seek_only_clear>/<size>/<filter_text>/<folder>')
def set_filter(label, value, seek_label, seek_only_clear, size, filter_text, folder):
    resp = jsonify(dbs.set_filter(label, value, seek_label, seek_only_clear, size, filter_text, folder))
    return resp  # ({'list': resp, 'len': len(resp)})


@app.route('/get_label_value_on_image/<label>/<im>')
def get_label_value_on_image(label, im):
    return jsonify(dbs.get_label_value_on_image(label, im))


"""
    Create your Model based REST API::

    class MyModelApi(ModelRestApi):
        datamodel = SQLAInterface(MyModel)

    appbuilder.add_api(MyModelApi)


    Create your Views::


    class MyModelView(ModelView):
        datamodel = SQLAInterface(MyModel)


    Next, register your Views::


    appbuilder.add_view(
        MyModelView,
        "My View",
        icon="fa-folder-open-o",
        category="My Category",
        category_icon='fa-envelope'
    )
"""

"""
    Application wide 404 error handler
"""

#
# @appbuilder.app.errorhandler(404)
# def page_not_found(e):
#     return (
#         render_template(
#             "404.html", base_template=appbuilder.base_template, appbuilder=appbuilder
#         ),
#         404,
#     )

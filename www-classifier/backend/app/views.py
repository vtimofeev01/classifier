from flask import send_from_directory, send_file, jsonify

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


@app.store_label("/label_<label>")
def store_label(label):
    return dbs.store_label(label)


@app.route('/image/<nm>')
def get_image_by_id(nm):
    return send_file(dbs.image(nm), mimetype='image/gif')


@app.route('/set_value/<im>/<label>/<code>')
def set_value(im, label, code):
    return dbs.set_value(im, label, code)


@app.route('/set_filter/<label>/<value>/<seek_label>')
def set_filter(label, value, seek_label):
    return dbs.set_filter(label, value, seek_label)


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

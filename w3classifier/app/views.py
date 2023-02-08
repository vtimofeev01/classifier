import datetime

from flask import send_from_directory, send_file, jsonify, Response, render_template, session, request, redirect, \
    url_for

F_FOLDER = 'folder'
F_LABEL = 'f_label'
F_LABEL_VALUE = 'f_label_value'
F_SIZE = 'f_size'
F_SIZES = ["height", "up", "low", "small"]
F_CLEAR = 'f_seek_only_clear'
ACT_MAIN_LABEL = 'act_main_label'
F_FAVORITES = 'f_favorites'
ACT_TO_FAVORITES = 'act_to_favorites'
ACT_SEEK = 'act_seek'
ACT_STORE = 'act_store'

from . import app, dbs


def E(v):
    return None if v == 'ALL' else v


@app.after_request
def add_header(r):
    """
    Add headers to both force latest IE rendering engine or Chrome Frame,
    and also to cache the rendered page for 10 minutes.
    """
    r.headers["Cache-Control"] = "no-cache, no-store, must-revalidate"
    r.headers["Pragma"] = "no-cache"
    r.headers["Expires"] = "0"
    r.headers['Cache-Control'] = 'public, max-age=0'
    return r


# Path for our main Svelte page
@app.route("/")
def base():
    args = request.args
    print(f'[integrity dbs] main_reid={len(dbs.main_reid)} == main={len(dbs.main)}')
    if 'command' in args:
        if F_FOLDER == args.get('command'):
            session[F_FOLDER] = E(args.get('value'))
            print(f'folder set to {session[F_FOLDER]}')
        if F_LABEL == args.get('command'):
            session[F_LABEL] = E(args.get('value'))
            print(f'filter label set to {session[F_LABEL]}')
            if session[F_LABEL] is None:
                session[F_LABEL_VALUE] = None
        if F_LABEL_VALUE == args.get('command'):
            session[F_LABEL_VALUE] = E(args.get('value'))
            print(f'filter label {session[F_LABEL]} by value {session[F_LABEL_VALUE]}')
        if F_SIZE == args.get('command'):
            session[F_SIZE] = E(args.get('value'))
            print(f'size id set to  {session[F_SIZE]}')
        if F_CLEAR == args.get('command'):
            session[F_CLEAR] = E(args.get('value'))
            print(f'clear is set to  {session[F_CLEAR]}')
        if F_FAVORITES == args.get('command'):
            session[F_FAVORITES] = E(args.get('value'))
            print(f'clear id set to  {session[F_FAVORITES]}')
        if ACT_MAIN_LABEL == args.get('command'):
            session[ACT_MAIN_LABEL] = E(args.get('value'))
            print(f'edit is set to  {session[ACT_MAIN_LABEL]}')
        dbs.return_filtered(label=session[ACT_MAIN_LABEL],
                            seek_label=session[F_LABEL],
                            seek_value=session[F_LABEL_VALUE],
                            seek_only_clear=session[F_CLEAR],
                            size=session[F_SIZE],
                            folder=session[F_FOLDER],
                            favorites=session[F_FAVORITES]
                            )
        session['index'] = int(0)
        session['count'] = int(dbs.filter.sum())
        return redirect(url_for("base"))
    if 'action' in args:
        print(f'action:{args["action"]}')
        if ACT_SEEK == args.get('action'):
            session['index'] += int(args.get('value'))
        if ACT_STORE == args.get('action') and session[ACT_MAIN_LABEL]:
            dbs.store_label(session[ACT_MAIN_LABEL])
            session[ACT_STORE] = f"{datetime.datetime.now():%d-%m-%y %H:%M:%S}"
        if ACT_TO_FAVORITES == args.get('action'):
            im_ix = session['index']
            im_name = dbs.navigation[im_ix]
            print(dbs.main)
            dbs.main.at[im_name,'favorites'] = not dbs.main.at[im_name, 'favorites']
            print(f'{im_name} id set to favorites: {session[F_FAVORITES]}')
        return redirect(url_for("base"))

    im_ix = session['index']
    if im_ix >= len(dbs.navigation):
        print(f"im_ix={im_ix} len(dbs.navigation)={len(dbs.navigation)}")
        return render_template('main.html', dbs=dbs, image=None, icons={})

    im_name = dbs.navigation[im_ix]
    image = dbs.main.loc[im_name]

    icons = dbs.return_label_value_on_image(
        session[ACT_MAIN_LABEL],
        image_name=im_name) if session[ACT_MAIN_LABEL] else {}

    label_value = dbs.main[session[ACT_MAIN_LABEL]][im_name] if session[ACT_MAIN_LABEL] else ""
    print(f"label_value={label_value}")

    return render_template('main.html', dbs=dbs, image_name=im_name, image=image, icons=icons, cur_value=label_value)


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

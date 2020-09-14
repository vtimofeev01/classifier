
(function(l, r) { if (l.getElementById('livereloadscript')) return; r = l.createElement('script'); r.async = 1; r.src = '//' + (window.location.host || 'localhost').split(':')[0] + ':35729/livereload.js?snipver=1'; r.id = 'livereloadscript'; l.getElementsByTagName('head')[0].appendChild(r) })(window.document);
var app = (function () {
    'use strict';

    function noop() { }
    function assign(tar, src) {
        // @ts-ignore
        for (const k in src)
            tar[k] = src[k];
        return tar;
    }
    function add_location(element, file, line, column, char) {
        element.__svelte_meta = {
            loc: { file, line, column, char }
        };
    }
    function run(fn) {
        return fn();
    }
    function blank_object() {
        return Object.create(null);
    }
    function run_all(fns) {
        fns.forEach(run);
    }
    function is_function(thing) {
        return typeof thing === 'function';
    }
    function safe_not_equal(a, b) {
        return a != a ? b == b : a !== b || ((a && typeof a === 'object') || typeof a === 'function');
    }
    function is_empty(obj) {
        return Object.keys(obj).length === 0;
    }
    function create_slot(definition, ctx, $$scope, fn) {
        if (definition) {
            const slot_ctx = get_slot_context(definition, ctx, $$scope, fn);
            return definition[0](slot_ctx);
        }
    }
    function get_slot_context(definition, ctx, $$scope, fn) {
        return definition[1] && fn
            ? assign($$scope.ctx.slice(), definition[1](fn(ctx)))
            : $$scope.ctx;
    }
    function get_slot_changes(definition, $$scope, dirty, fn) {
        if (definition[2] && fn) {
            const lets = definition[2](fn(dirty));
            if ($$scope.dirty === undefined) {
                return lets;
            }
            if (typeof lets === 'object') {
                const merged = [];
                const len = Math.max($$scope.dirty.length, lets.length);
                for (let i = 0; i < len; i += 1) {
                    merged[i] = $$scope.dirty[i] | lets[i];
                }
                return merged;
            }
            return $$scope.dirty | lets;
        }
        return $$scope.dirty;
    }
    function update_slot(slot, slot_definition, ctx, $$scope, dirty, get_slot_changes_fn, get_slot_context_fn) {
        const slot_changes = get_slot_changes(slot_definition, $$scope, dirty, get_slot_changes_fn);
        if (slot_changes) {
            const slot_context = get_slot_context(slot_definition, ctx, $$scope, get_slot_context_fn);
            slot.p(slot_context, slot_changes);
        }
    }
    function null_to_empty(value) {
        return value == null ? '' : value;
    }

    function append(target, node) {
        target.appendChild(node);
    }
    function insert(target, node, anchor) {
        target.insertBefore(node, anchor || null);
    }
    function detach(node) {
        node.parentNode.removeChild(node);
    }
    function destroy_each(iterations, detaching) {
        for (let i = 0; i < iterations.length; i += 1) {
            if (iterations[i])
                iterations[i].d(detaching);
        }
    }
    function element(name) {
        return document.createElement(name);
    }
    function text(data) {
        return document.createTextNode(data);
    }
    function space() {
        return text(' ');
    }
    function empty() {
        return text('');
    }
    function listen(node, event, handler, options) {
        node.addEventListener(event, handler, options);
        return () => node.removeEventListener(event, handler, options);
    }
    function attr(node, attribute, value) {
        if (value == null)
            node.removeAttribute(attribute);
        else if (node.getAttribute(attribute) !== value)
            node.setAttribute(attribute, value);
    }
    function children(element) {
        return Array.from(element.childNodes);
    }
    function toggle_class(element, name, toggle) {
        element.classList[toggle ? 'add' : 'remove'](name);
    }
    function custom_event(type, detail) {
        const e = document.createEvent('CustomEvent');
        e.initCustomEvent(type, false, false, detail);
        return e;
    }

    let current_component;
    function set_current_component(component) {
        current_component = component;
    }
    function get_current_component() {
        if (!current_component)
            throw new Error(`Function called outside component initialization`);
        return current_component;
    }
    function onMount(fn) {
        get_current_component().$$.on_mount.push(fn);
    }

    const dirty_components = [];
    const binding_callbacks = [];
    const render_callbacks = [];
    const flush_callbacks = [];
    const resolved_promise = Promise.resolve();
    let update_scheduled = false;
    function schedule_update() {
        if (!update_scheduled) {
            update_scheduled = true;
            resolved_promise.then(flush);
        }
    }
    function add_render_callback(fn) {
        render_callbacks.push(fn);
    }
    let flushing = false;
    const seen_callbacks = new Set();
    function flush() {
        if (flushing)
            return;
        flushing = true;
        do {
            // first, call beforeUpdate functions
            // and update components
            for (let i = 0; i < dirty_components.length; i += 1) {
                const component = dirty_components[i];
                set_current_component(component);
                update(component.$$);
            }
            dirty_components.length = 0;
            while (binding_callbacks.length)
                binding_callbacks.pop()();
            // then, once components are updated, call
            // afterUpdate functions. This may cause
            // subsequent updates...
            for (let i = 0; i < render_callbacks.length; i += 1) {
                const callback = render_callbacks[i];
                if (!seen_callbacks.has(callback)) {
                    // ...so guard against infinite loops
                    seen_callbacks.add(callback);
                    callback();
                }
            }
            render_callbacks.length = 0;
        } while (dirty_components.length);
        while (flush_callbacks.length) {
            flush_callbacks.pop()();
        }
        update_scheduled = false;
        flushing = false;
        seen_callbacks.clear();
    }
    function update($$) {
        if ($$.fragment !== null) {
            $$.update();
            run_all($$.before_update);
            const dirty = $$.dirty;
            $$.dirty = [-1];
            $$.fragment && $$.fragment.p($$.ctx, dirty);
            $$.after_update.forEach(add_render_callback);
        }
    }
    const outroing = new Set();
    let outros;
    function transition_in(block, local) {
        if (block && block.i) {
            outroing.delete(block);
            block.i(local);
        }
    }
    function transition_out(block, local, detach, callback) {
        if (block && block.o) {
            if (outroing.has(block))
                return;
            outroing.add(block);
            outros.c.push(() => {
                outroing.delete(block);
                if (callback) {
                    if (detach)
                        block.d(1);
                    callback();
                }
            });
            block.o(local);
        }
    }

    const globals = (typeof window !== 'undefined'
        ? window
        : typeof globalThis !== 'undefined'
            ? globalThis
            : global);
    function create_component(block) {
        block && block.c();
    }
    function mount_component(component, target, anchor) {
        const { fragment, on_mount, on_destroy, after_update } = component.$$;
        fragment && fragment.m(target, anchor);
        // onMount happens before the initial afterUpdate
        add_render_callback(() => {
            const new_on_destroy = on_mount.map(run).filter(is_function);
            if (on_destroy) {
                on_destroy.push(...new_on_destroy);
            }
            else {
                // Edge case - component was destroyed immediately,
                // most likely as a result of a binding initialising
                run_all(new_on_destroy);
            }
            component.$$.on_mount = [];
        });
        after_update.forEach(add_render_callback);
    }
    function destroy_component(component, detaching) {
        const $$ = component.$$;
        if ($$.fragment !== null) {
            run_all($$.on_destroy);
            $$.fragment && $$.fragment.d(detaching);
            // TODO null out other refs, including component.$$ (but need to
            // preserve final state?)
            $$.on_destroy = $$.fragment = null;
            $$.ctx = [];
        }
    }
    function make_dirty(component, i) {
        if (component.$$.dirty[0] === -1) {
            dirty_components.push(component);
            schedule_update();
            component.$$.dirty.fill(0);
        }
        component.$$.dirty[(i / 31) | 0] |= (1 << (i % 31));
    }
    function init(component, options, instance, create_fragment, not_equal, props, dirty = [-1]) {
        const parent_component = current_component;
        set_current_component(component);
        const prop_values = options.props || {};
        const $$ = component.$$ = {
            fragment: null,
            ctx: null,
            // state
            props,
            update: noop,
            not_equal,
            bound: blank_object(),
            // lifecycle
            on_mount: [],
            on_destroy: [],
            before_update: [],
            after_update: [],
            context: new Map(parent_component ? parent_component.$$.context : []),
            // everything else
            callbacks: blank_object(),
            dirty,
            skip_bound: false
        };
        let ready = false;
        $$.ctx = instance
            ? instance(component, prop_values, (i, ret, ...rest) => {
                const value = rest.length ? rest[0] : ret;
                if ($$.ctx && not_equal($$.ctx[i], $$.ctx[i] = value)) {
                    if (!$$.skip_bound && $$.bound[i])
                        $$.bound[i](value);
                    if (ready)
                        make_dirty(component, i);
                }
                return ret;
            })
            : [];
        $$.update();
        ready = true;
        run_all($$.before_update);
        // `false` as a special case of no DOM component
        $$.fragment = create_fragment ? create_fragment($$.ctx) : false;
        if (options.target) {
            if (options.hydrate) {
                const nodes = children(options.target);
                // eslint-disable-next-line @typescript-eslint/no-non-null-assertion
                $$.fragment && $$.fragment.l(nodes);
                nodes.forEach(detach);
            }
            else {
                // eslint-disable-next-line @typescript-eslint/no-non-null-assertion
                $$.fragment && $$.fragment.c();
            }
            if (options.intro)
                transition_in(component.$$.fragment);
            mount_component(component, options.target, options.anchor);
            flush();
        }
        set_current_component(parent_component);
    }
    class SvelteComponent {
        $destroy() {
            destroy_component(this, 1);
            this.$destroy = noop;
        }
        $on(type, callback) {
            const callbacks = (this.$$.callbacks[type] || (this.$$.callbacks[type] = []));
            callbacks.push(callback);
            return () => {
                const index = callbacks.indexOf(callback);
                if (index !== -1)
                    callbacks.splice(index, 1);
            };
        }
        $set($$props) {
            if (this.$$set && !is_empty($$props)) {
                this.$$.skip_bound = true;
                this.$$set($$props);
                this.$$.skip_bound = false;
            }
        }
    }

    function dispatch_dev(type, detail) {
        document.dispatchEvent(custom_event(type, Object.assign({ version: '3.24.1' }, detail)));
    }
    function append_dev(target, node) {
        dispatch_dev("SvelteDOMInsert", { target, node });
        append(target, node);
    }
    function insert_dev(target, node, anchor) {
        dispatch_dev("SvelteDOMInsert", { target, node, anchor });
        insert(target, node, anchor);
    }
    function detach_dev(node) {
        dispatch_dev("SvelteDOMRemove", { node });
        detach(node);
    }
    function listen_dev(node, event, handler, options, has_prevent_default, has_stop_propagation) {
        const modifiers = options === true ? ["capture"] : options ? Array.from(Object.keys(options)) : [];
        if (has_prevent_default)
            modifiers.push('preventDefault');
        if (has_stop_propagation)
            modifiers.push('stopPropagation');
        dispatch_dev("SvelteDOMAddEventListener", { node, event, handler, modifiers });
        const dispose = listen(node, event, handler, options);
        return () => {
            dispatch_dev("SvelteDOMRemoveEventListener", { node, event, handler, modifiers });
            dispose();
        };
    }
    function attr_dev(node, attribute, value) {
        attr(node, attribute, value);
        if (value == null)
            dispatch_dev("SvelteDOMRemoveAttribute", { node, attribute });
        else
            dispatch_dev("SvelteDOMSetAttribute", { node, attribute, value });
    }
    function prop_dev(node, property, value) {
        node[property] = value;
        dispatch_dev("SvelteDOMSetProperty", { node, property, value });
    }
    function set_data_dev(text, data) {
        data = '' + data;
        if (text.wholeText === data)
            return;
        dispatch_dev("SvelteDOMSetData", { node: text, data });
        text.data = data;
    }
    function validate_each_argument(arg) {
        if (typeof arg !== 'string' && !(arg && typeof arg === 'object' && 'length' in arg)) {
            let msg = '{#each} only iterates over array-like objects.';
            if (typeof Symbol === 'function' && arg && Symbol.iterator in arg) {
                msg += ' You can use a spread to convert this iterable into an array.';
            }
            throw new Error(msg);
        }
    }
    function validate_slots(name, slot, keys) {
        for (const slot_key of Object.keys(slot)) {
            if (!~keys.indexOf(slot_key)) {
                console.warn(`<${name}> received an unexpected slot "${slot_key}".`);
            }
        }
    }
    class SvelteComponentDev extends SvelteComponent {
        constructor(options) {
            if (!options || (!options.target && !options.$$inline)) {
                throw new Error(`'target' is a required option`);
            }
            super();
        }
        $destroy() {
            super.$destroy();
            this.$destroy = () => {
                console.warn(`Component was already destroyed`); // eslint-disable-line no-console
            };
        }
        $capture_state() { }
        $inject_state() { }
    }

    /* src/Box.svelte generated by Svelte v3.24.1 */

    const file = "src/Box.svelte";

    function create_fragment(ctx) {
    	let div;
    	let div_class_value;
    	let current;
    	const default_slot_template = /*$$slots*/ ctx[2].default;
    	const default_slot = create_slot(default_slot_template, ctx, /*$$scope*/ ctx[1], null);

    	const block = {
    		c: function create() {
    			div = element("div");
    			if (default_slot) default_slot.c();
    			attr_dev(div, "class", div_class_value = "" + (null_to_empty(/*cls*/ ctx[0]) + " svelte-hbgish"));
    			add_location(div, file, 50, 0, 840);
    		},
    		l: function claim(nodes) {
    			throw new Error("options.hydrate only works if the component was compiled with the `hydratable: true` option");
    		},
    		m: function mount(target, anchor) {
    			insert_dev(target, div, anchor);

    			if (default_slot) {
    				default_slot.m(div, null);
    			}

    			current = true;
    		},
    		p: function update(ctx, [dirty]) {
    			if (default_slot) {
    				if (default_slot.p && dirty & /*$$scope*/ 2) {
    					update_slot(default_slot, default_slot_template, ctx, /*$$scope*/ ctx[1], dirty, null, null);
    				}
    			}

    			if (!current || dirty & /*cls*/ 1 && div_class_value !== (div_class_value = "" + (null_to_empty(/*cls*/ ctx[0]) + " svelte-hbgish"))) {
    				attr_dev(div, "class", div_class_value);
    			}
    		},
    		i: function intro(local) {
    			if (current) return;
    			transition_in(default_slot, local);
    			current = true;
    		},
    		o: function outro(local) {
    			transition_out(default_slot, local);
    			current = false;
    		},
    		d: function destroy(detaching) {
    			if (detaching) detach_dev(div);
    			if (default_slot) default_slot.d(detaching);
    		}
    	};

    	dispatch_dev("SvelteRegisterBlock", {
    		block,
    		id: create_fragment.name,
    		type: "component",
    		source: "",
    		ctx
    	});

    	return block;
    }

    function instance($$self, $$props, $$invalidate) {
    	let { cls = "box" } = $$props;
    	const writable_props = ["cls"];

    	Object.keys($$props).forEach(key => {
    		if (!~writable_props.indexOf(key) && key.slice(0, 2) !== "$$") console.warn(`<Box> was created with unknown prop '${key}'`);
    	});

    	let { $$slots = {}, $$scope } = $$props;
    	validate_slots("Box", $$slots, ['default']);

    	$$self.$$set = $$props => {
    		if ("cls" in $$props) $$invalidate(0, cls = $$props.cls);
    		if ("$$scope" in $$props) $$invalidate(1, $$scope = $$props.$$scope);
    	};

    	$$self.$capture_state = () => ({ cls });

    	$$self.$inject_state = $$props => {
    		if ("cls" in $$props) $$invalidate(0, cls = $$props.cls);
    	};

    	if ($$props && "$$inject" in $$props) {
    		$$self.$inject_state($$props.$$inject);
    	}

    	return [cls, $$scope, $$slots];
    }

    class Box extends SvelteComponentDev {
    	constructor(options) {
    		super(options);
    		init(this, options, instance, create_fragment, safe_not_equal, { cls: 0 });

    		dispatch_dev("SvelteRegisterComponent", {
    			component: this,
    			tagName: "Box",
    			options,
    			id: create_fragment.name
    		});
    	}

    	get cls() {
    		throw new Error("<Box>: Props cannot be read directly from the component instance unless compiling with 'accessors: true' or '<svelte:options accessors/>'");
    	}

    	set cls(value) {
    		throw new Error("<Box>: Props cannot be set directly on the component instance unless compiling with 'accessors: true' or '<svelte:options accessors/>'");
    	}
    }

    /* src/App.svelte generated by Svelte v3.24.1 */

    const { console: console_1 } = globals;
    const file$1 = "src/App.svelte";

    function get_each_context(ctx, list, i) {
    	const child_ctx = ctx.slice();
    	child_ctx[47] = list[i];
    	return child_ctx;
    }

    function get_each_context_1(ctx, list, i) {
    	const child_ctx = ctx.slice();
    	child_ctx[47] = list[i];
    	return child_ctx;
    }

    function get_each_context_2(ctx, list, i) {
    	const child_ctx = ctx.slice();
    	child_ctx[52] = list[i];
    	return child_ctx;
    }

    function get_each_context_3(ctx, list, i) {
    	const child_ctx = ctx.slice();
    	child_ctx[55] = list[i];
    	child_ctx[57] = i;
    	return child_ctx;
    }

    function get_each_context_4(ctx, list, i) {
    	const child_ctx = ctx.slice();
    	child_ctx[52] = list[i];
    	return child_ctx;
    }

    // (202:4) {#each all_labels as ilabel}
    function create_each_block_4(ctx) {
    	let button;
    	let t_value = /*ilabel*/ ctx[52] + "";
    	let t;
    	let mounted;
    	let dispose;

    	function click_handler_1(...args) {
    		return /*click_handler_1*/ ctx[21](/*ilabel*/ ctx[52], ...args);
    	}

    	const block = {
    		c: function create() {
    			button = element("button");
    			t = text(t_value);
    			attr_dev(button, "class", "label svelte-1y02p24");
    			toggle_class(button, "selected", /*seek_label*/ ctx[6] === /*ilabel*/ ctx[52]);
    			add_location(button, file$1, 202, 8, 5307);
    		},
    		m: function mount(target, anchor) {
    			insert_dev(target, button, anchor);
    			append_dev(button, t);

    			if (!mounted) {
    				dispose = listen_dev(button, "click", click_handler_1, false, false, false);
    				mounted = true;
    			}
    		},
    		p: function update(new_ctx, dirty) {
    			ctx = new_ctx;
    			if (dirty[0] & /*all_labels*/ 4096 && t_value !== (t_value = /*ilabel*/ ctx[52] + "")) set_data_dev(t, t_value);

    			if (dirty[0] & /*seek_label, all_labels*/ 4160) {
    				toggle_class(button, "selected", /*seek_label*/ ctx[6] === /*ilabel*/ ctx[52]);
    			}
    		},
    		d: function destroy(detaching) {
    			if (detaching) detach_dev(button);
    			mounted = false;
    			dispose();
    		}
    	};

    	dispatch_dev("SvelteRegisterBlock", {
    		block,
    		id: create_each_block_4.name,
    		type: "each",
    		source: "(202:4) {#each all_labels as ilabel}",
    		ctx
    	});

    	return block;
    }

    // (199:0) <Box>
    function create_default_slot_2(ctx) {
    	let h3;
    	let t1;
    	let button0;
    	let t2;
    	let t3;
    	let t4;
    	let button0_disabled_value;
    	let t5;
    	let t6;
    	let button1;
    	let t8;
    	let button2;
    	let t10;
    	let br;
    	let mounted;
    	let dispose;
    	let each_value_4 = /*all_labels*/ ctx[12];
    	validate_each_argument(each_value_4);
    	let each_blocks = [];

    	for (let i = 0; i < each_value_4.length; i += 1) {
    		each_blocks[i] = create_each_block_4(get_each_context_4(ctx, each_value_4, i));
    	}

    	const block = {
    		c: function create() {
    			h3 = element("h3");
    			h3.textContent = "select label to edit";
    			t1 = space();
    			button0 = element("button");
    			t2 = text(/*storetext*/ ctx[13]);
    			t3 = text(":");
    			t4 = text(/*seek_label*/ ctx[6]);
    			t5 = space();

    			for (let i = 0; i < each_blocks.length; i += 1) {
    				each_blocks[i].c();
    			}

    			t6 = space();
    			button1 = element("button");
    			button1.textContent = "new";
    			t8 = space();
    			button2 = element("button");
    			button2.textContent = "all";
    			t10 = space();
    			br = element("br");
    			add_location(h3, file$1, 199, 4, 5120);
    			button0.disabled = button0_disabled_value = /*seek_label*/ ctx[6] === "none";
    			add_location(button0, file$1, 200, 4, 5155);
    			attr_dev(button1, "class", "svelte-1y02p24");
    			toggle_class(button1, "selected", /*seek_only_clear*/ ctx[7] === "yes");
    			add_location(button1, file$1, 205, 4, 5472);
    			attr_dev(button2, "class", "svelte-1y02p24");
    			toggle_class(button2, "selected", /*seek_only_clear*/ ctx[7] === "no");
    			add_location(button2, file$1, 207, 4, 5603);
    			add_location(br, file$1, 209, 4, 5732);
    		},
    		m: function mount(target, anchor) {
    			insert_dev(target, h3, anchor);
    			insert_dev(target, t1, anchor);
    			insert_dev(target, button0, anchor);
    			append_dev(button0, t2);
    			append_dev(button0, t3);
    			append_dev(button0, t4);
    			insert_dev(target, t5, anchor);

    			for (let i = 0; i < each_blocks.length; i += 1) {
    				each_blocks[i].m(target, anchor);
    			}

    			insert_dev(target, t6, anchor);
    			insert_dev(target, button1, anchor);
    			insert_dev(target, t8, anchor);
    			insert_dev(target, button2, anchor);
    			insert_dev(target, t10, anchor);
    			insert_dev(target, br, anchor);

    			if (!mounted) {
    				dispose = [
    					listen_dev(button0, "click", /*click_handler*/ ctx[20], false, false, false),
    					listen_dev(button1, "click", /*click_handler_2*/ ctx[22], false, false, false),
    					listen_dev(button2, "click", /*click_handler_3*/ ctx[23], false, false, false)
    				];

    				mounted = true;
    			}
    		},
    		p: function update(ctx, dirty) {
    			if (dirty[0] & /*storetext*/ 8192) set_data_dev(t2, /*storetext*/ ctx[13]);
    			if (dirty[0] & /*seek_label*/ 64) set_data_dev(t4, /*seek_label*/ ctx[6]);

    			if (dirty[0] & /*seek_label*/ 64 && button0_disabled_value !== (button0_disabled_value = /*seek_label*/ ctx[6] === "none")) {
    				prop_dev(button0, "disabled", button0_disabled_value);
    			}

    			if (dirty[0] & /*seek_label, all_labels, loadData*/ 135232) {
    				each_value_4 = /*all_labels*/ ctx[12];
    				validate_each_argument(each_value_4);
    				let i;

    				for (i = 0; i < each_value_4.length; i += 1) {
    					const child_ctx = get_each_context_4(ctx, each_value_4, i);

    					if (each_blocks[i]) {
    						each_blocks[i].p(child_ctx, dirty);
    					} else {
    						each_blocks[i] = create_each_block_4(child_ctx);
    						each_blocks[i].c();
    						each_blocks[i].m(t6.parentNode, t6);
    					}
    				}

    				for (; i < each_blocks.length; i += 1) {
    					each_blocks[i].d(1);
    				}

    				each_blocks.length = each_value_4.length;
    			}

    			if (dirty[0] & /*seek_only_clear*/ 128) {
    				toggle_class(button1, "selected", /*seek_only_clear*/ ctx[7] === "yes");
    			}

    			if (dirty[0] & /*seek_only_clear*/ 128) {
    				toggle_class(button2, "selected", /*seek_only_clear*/ ctx[7] === "no");
    			}
    		},
    		d: function destroy(detaching) {
    			if (detaching) detach_dev(h3);
    			if (detaching) detach_dev(t1);
    			if (detaching) detach_dev(button0);
    			if (detaching) detach_dev(t5);
    			destroy_each(each_blocks, detaching);
    			if (detaching) detach_dev(t6);
    			if (detaching) detach_dev(button1);
    			if (detaching) detach_dev(t8);
    			if (detaching) detach_dev(button2);
    			if (detaching) detach_dev(t10);
    			if (detaching) detach_dev(br);
    			mounted = false;
    			run_all(dispose);
    		}
    	};

    	dispatch_dev("SvelteRegisterBlock", {
    		block,
    		id: create_default_slot_2.name,
    		type: "slot",
    		source: "(199:0) <Box>",
    		ctx
    	});

    	return block;
    }

    // (218:16) {#if images[index]=undefined}
    function create_if_block_6(ctx) {
    	let img;
    	let img_src_value;
    	let img_alt_value;

    	const block = {
    		c: function create() {
    			img = element("img");
    			if (img.src !== (img_src_value = "" + (/*source_server*/ ctx[14] + "/marked_image/" + /*images*/ ctx[11][/*index*/ ctx[10]]))) attr_dev(img, "src", img_src_value);
    			attr_dev(img, "alt", img_alt_value = /*images*/ ctx[11][/*index*/ ctx[10]]);
    			attr_dev(img, "class", "svelte-1y02p24");
    			add_location(img, file$1, 218, 16, 5856);
    		},
    		m: function mount(target, anchor) {
    			insert_dev(target, img, anchor);
    		},
    		p: function update(ctx, dirty) {
    			if (dirty[0] & /*images, index*/ 3072 && img.src !== (img_src_value = "" + (/*source_server*/ ctx[14] + "/marked_image/" + /*images*/ ctx[11][/*index*/ ctx[10]]))) {
    				attr_dev(img, "src", img_src_value);
    			}

    			if (dirty[0] & /*images, index*/ 3072 && img_alt_value !== (img_alt_value = /*images*/ ctx[11][/*index*/ ctx[10]])) {
    				attr_dev(img, "alt", img_alt_value);
    			}
    		},
    		d: function destroy(detaching) {
    			if (detaching) detach_dev(img);
    		}
    	};

    	dispatch_dev("SvelteRegisterBlock", {
    		block,
    		id: create_if_block_6.name,
    		type: "if",
    		source: "(218:16) {#if images[index]=undefined}",
    		ctx
    	});

    	return block;
    }

    // (222:16) {#each seekvalues as sv, nn}
    function create_each_block_3(ctx) {
    	let button;
    	let t0;
    	let t1;
    	let t2;
    	let t3_value = /*sv*/ ctx[55] + "";
    	let t3;
    	let br;
    	let t4;
    	let mounted;
    	let dispose;

    	function click_handler_4(...args) {
    		return /*click_handler_4*/ ctx[24](/*sv*/ ctx[55], ...args);
    	}

    	const block = {
    		c: function create() {
    			button = element("button");
    			t0 = text("(");
    			t1 = text(/*nn*/ ctx[57]);
    			t2 = text("\n                        ) ");
    			t3 = text(t3_value);
    			br = element("br");
    			t4 = space();
    			attr_dev(button, "class", "svelte-1y02p24");
    			toggle_class(button, "active", /*seekvalue*/ ctx[9] === /*sv*/ ctx[55]);
    			add_location(button, file$1, 222, 20, 6044);
    			add_location(br, file$1, 223, 39, 6184);
    		},
    		m: function mount(target, anchor) {
    			insert_dev(target, button, anchor);
    			append_dev(button, t0);
    			append_dev(button, t1);
    			append_dev(button, t2);
    			append_dev(button, t3);
    			insert_dev(target, br, anchor);
    			insert_dev(target, t4, anchor);

    			if (!mounted) {
    				dispose = listen_dev(button, "click", click_handler_4, false, false, false);
    				mounted = true;
    			}
    		},
    		p: function update(new_ctx, dirty) {
    			ctx = new_ctx;
    			if (dirty[0] & /*seekvalues*/ 256 && t3_value !== (t3_value = /*sv*/ ctx[55] + "")) set_data_dev(t3, t3_value);

    			if (dirty[0] & /*seekvalue, seekvalues*/ 768) {
    				toggle_class(button, "active", /*seekvalue*/ ctx[9] === /*sv*/ ctx[55]);
    			}
    		},
    		d: function destroy(detaching) {
    			if (detaching) detach_dev(button);
    			if (detaching) detach_dev(br);
    			if (detaching) detach_dev(t4);
    			mounted = false;
    			dispose();
    		}
    	};

    	dispatch_dev("SvelteRegisterBlock", {
    		block,
    		id: create_each_block_3.name,
    		type: "each",
    		source: "(222:16) {#each seekvalues as sv, nn}",
    		ctx
    	});

    	return block;
    }

    // (227:16) {#if filter_value === 'to_check'}
    function create_if_block_5(ctx) {
    	let br;
    	let t_value = /*filter_textes*/ ctx[5][/*index*/ ctx[10]] + "";
    	let t;

    	const block = {
    		c: function create() {
    			br = element("br");
    			t = text(t_value);
    			add_location(br, file$1, 227, 20, 6427);
    		},
    		m: function mount(target, anchor) {
    			insert_dev(target, br, anchor);
    			insert_dev(target, t, anchor);
    		},
    		p: function update(ctx, dirty) {
    			if (dirty[0] & /*filter_textes, index*/ 1056 && t_value !== (t_value = /*filter_textes*/ ctx[5][/*index*/ ctx[10]] + "")) set_data_dev(t, t_value);
    		},
    		d: function destroy(detaching) {
    			if (detaching) detach_dev(br);
    			if (detaching) detach_dev(t);
    		}
    	};

    	dispatch_dev("SvelteRegisterBlock", {
    		block,
    		id: create_if_block_5.name,
    		type: "if",
    		source: "(227:16) {#if filter_value === 'to_check'}",
    		ctx
    	});

    	return block;
    }

    // (236:4) {#if imslen > 100}
    function create_if_block_4(ctx) {
    	let button;
    	let t;
    	let button_disabled_value;
    	let mounted;
    	let dispose;

    	const block = {
    		c: function create() {
    			button = element("button");
    			t = text("-100");
    			button.disabled = button_disabled_value = /*index*/ ctx[10] < 100;
    			add_location(button, file$1, 236, 8, 6645);
    		},
    		m: function mount(target, anchor) {
    			insert_dev(target, button, anchor);
    			append_dev(button, t);

    			if (!mounted) {
    				dispose = listen_dev(button, "click", /*click_handler_7*/ ctx[27], false, false, false);
    				mounted = true;
    			}
    		},
    		p: function update(ctx, dirty) {
    			if (dirty[0] & /*index*/ 1024 && button_disabled_value !== (button_disabled_value = /*index*/ ctx[10] < 100)) {
    				prop_dev(button, "disabled", button_disabled_value);
    			}
    		},
    		d: function destroy(detaching) {
    			if (detaching) detach_dev(button);
    			mounted = false;
    			dispose();
    		}
    	};

    	dispatch_dev("SvelteRegisterBlock", {
    		block,
    		id: create_if_block_4.name,
    		type: "if",
    		source: "(236:4) {#if imslen > 100}",
    		ctx
    	});

    	return block;
    }

    // (239:4) {#if imslen > 1000}
    function create_if_block_3(ctx) {
    	let button;
    	let t;
    	let button_disabled_value;
    	let mounted;
    	let dispose;

    	const block = {
    		c: function create() {
    			button = element("button");
    			t = text("-1000");
    			button.disabled = button_disabled_value = /*index*/ ctx[10] < 1000;
    			add_location(button, file$1, 239, 8, 6779);
    		},
    		m: function mount(target, anchor) {
    			insert_dev(target, button, anchor);
    			append_dev(button, t);

    			if (!mounted) {
    				dispose = listen_dev(button, "click", /*click_handler_8*/ ctx[28], false, false, false);
    				mounted = true;
    			}
    		},
    		p: function update(ctx, dirty) {
    			if (dirty[0] & /*index*/ 1024 && button_disabled_value !== (button_disabled_value = /*index*/ ctx[10] < 1000)) {
    				prop_dev(button, "disabled", button_disabled_value);
    			}
    		},
    		d: function destroy(detaching) {
    			if (detaching) detach_dev(button);
    			mounted = false;
    			dispose();
    		}
    	};

    	dispatch_dev("SvelteRegisterBlock", {
    		block,
    		id: create_if_block_3.name,
    		type: "if",
    		source: "(239:4) {#if imslen > 1000}",
    		ctx
    	});

    	return block;
    }

    // (243:4) {#if imslen > 1000}
    function create_if_block_2(ctx) {
    	let button;
    	let t;
    	let button_disabled_value;
    	let mounted;
    	let dispose;

    	const block = {
    		c: function create() {
    			button = element("button");
    			t = text("+1000");
    			button.disabled = button_disabled_value = /*index*/ ctx[10] > /*imslen*/ ctx[0] - 1000;
    			add_location(button, file$1, 243, 8, 6939);
    		},
    		m: function mount(target, anchor) {
    			insert_dev(target, button, anchor);
    			append_dev(button, t);

    			if (!mounted) {
    				dispose = listen_dev(button, "click", /*click_handler_9*/ ctx[29], false, false, false);
    				mounted = true;
    			}
    		},
    		p: function update(ctx, dirty) {
    			if (dirty[0] & /*index, imslen*/ 1025 && button_disabled_value !== (button_disabled_value = /*index*/ ctx[10] > /*imslen*/ ctx[0] - 1000)) {
    				prop_dev(button, "disabled", button_disabled_value);
    			}
    		},
    		d: function destroy(detaching) {
    			if (detaching) detach_dev(button);
    			mounted = false;
    			dispose();
    		}
    	};

    	dispatch_dev("SvelteRegisterBlock", {
    		block,
    		id: create_if_block_2.name,
    		type: "if",
    		source: "(243:4) {#if imslen > 1000}",
    		ctx
    	});

    	return block;
    }

    // (246:4) {#if imslen > 100}
    function create_if_block_1(ctx) {
    	let button;
    	let t;
    	let button_disabled_value;
    	let mounted;
    	let dispose;

    	const block = {
    		c: function create() {
    			button = element("button");
    			t = text("+100");
    			button.disabled = button_disabled_value = /*index*/ ctx[10] > /*imslen*/ ctx[0] - 100;
    			add_location(button, file$1, 246, 8, 7084);
    		},
    		m: function mount(target, anchor) {
    			insert_dev(target, button, anchor);
    			append_dev(button, t);

    			if (!mounted) {
    				dispose = listen_dev(button, "click", /*click_handler_10*/ ctx[30], false, false, false);
    				mounted = true;
    			}
    		},
    		p: function update(ctx, dirty) {
    			if (dirty[0] & /*index, imslen*/ 1025 && button_disabled_value !== (button_disabled_value = /*index*/ ctx[10] > /*imslen*/ ctx[0] - 100)) {
    				prop_dev(button, "disabled", button_disabled_value);
    			}
    		},
    		d: function destroy(detaching) {
    			if (detaching) detach_dev(button);
    			mounted = false;
    			dispose();
    		}
    	};

    	dispatch_dev("SvelteRegisterBlock", {
    		block,
    		id: create_if_block_1.name,
    		type: "if",
    		source: "(246:4) {#if imslen > 100}",
    		ctx
    	});

    	return block;
    }

    // (214:0) <Box>
    function create_default_slot_1(ctx) {
    	let table;
    	let tr;
    	let td0;
    	let br0;
    	let t0;
    	let td1;
    	let br1;
    	let t1;
    	let button0;
    	let br2;
    	let t3;
    	let t4;
    	let button1;
    	let t5;
    	let button1_disabled_value;
    	let t6;
    	let t7;
    	let t8;
    	let t9;
    	let t10;
    	let t11;
    	let t12;
    	let t13;
    	let t14;
    	let button2;
    	let t15;
    	let button2_disabled_value;
    	let mounted;
    	let dispose;
    	let if_block0 = (/*images*/ ctx[11][/*index*/ ctx[10]] = undefined) && create_if_block_6(ctx);
    	let each_value_3 = /*seekvalues*/ ctx[8];
    	validate_each_argument(each_value_3);
    	let each_blocks = [];

    	for (let i = 0; i < each_value_3.length; i += 1) {
    		each_blocks[i] = create_each_block_3(get_each_context_3(ctx, each_value_3, i));
    	}

    	let if_block1 = /*filter_value*/ ctx[2] === "to_check" && create_if_block_5(ctx);
    	let if_block2 = /*imslen*/ ctx[0] > 100 && create_if_block_4(ctx);
    	let if_block3 = /*imslen*/ ctx[0] > 1000 && create_if_block_3(ctx);
    	let if_block4 = /*imslen*/ ctx[0] > 1000 && create_if_block_2(ctx);
    	let if_block5 = /*imslen*/ ctx[0] > 100 && create_if_block_1(ctx);

    	const block = {
    		c: function create() {
    			table = element("table");
    			tr = element("tr");
    			td0 = element("td");
    			if (if_block0) if_block0.c();
    			br0 = element("br");
    			t0 = space();
    			td1 = element("td");

    			for (let i = 0; i < each_blocks.length; i += 1) {
    				each_blocks[i].c();
    			}

    			br1 = element("br");
    			t1 = space();
    			button0 = element("button");
    			button0.textContent = "Del";
    			br2 = element("br");
    			t3 = space();
    			if (if_block1) if_block1.c();
    			t4 = space();
    			button1 = element("button");
    			t5 = text("prev");
    			t6 = space();
    			if (if_block2) if_block2.c();
    			t7 = space();
    			if (if_block3) if_block3.c();
    			t8 = space();
    			t9 = text(/*index*/ ctx[10]);
    			t10 = text(" / ");
    			t11 = text(/*imslen*/ ctx[0]);
    			t12 = space();
    			if (if_block4) if_block4.c();
    			t13 = space();
    			if (if_block5) if_block5.c();
    			t14 = space();
    			button2 = element("button");
    			t15 = text("next");
    			add_location(br0, file$1, 218, 99, 5939);
    			attr_dev(td0, "class", "svelte-1y02p24");
    			add_location(td0, file$1, 216, 12, 5789);
    			add_location(br1, file$1, 224, 23, 6212);
    			attr_dev(button0, "class", "svelte-1y02p24");
    			toggle_class(button0, "active", /*seekvalue*/ ctx[9] === "DELETE");
    			add_location(button0, file$1, 225, 16, 6233);
    			add_location(br2, file$1, 225, 135, 6352);
    			attr_dev(td1, "class", "svelte-1y02p24");
    			add_location(td1, file$1, 220, 12, 5974);
    			attr_dev(tr, "class", "svelte-1y02p24");
    			add_location(tr, file$1, 215, 8, 5772);
    			attr_dev(table, "class", "svelte-1y02p24");
    			add_location(table, file$1, 214, 4, 5756);
    			button1.disabled = button1_disabled_value = /*index*/ ctx[10] === 0;
    			add_location(button1, file$1, 234, 4, 6527);
    			button2.disabled = button2_disabled_value = /*index*/ ctx[10] === /*imslen*/ ctx[0];
    			add_location(button2, file$1, 248, 4, 7199);
    		},
    		m: function mount(target, anchor) {
    			insert_dev(target, table, anchor);
    			append_dev(table, tr);
    			append_dev(tr, td0);
    			if (if_block0) if_block0.m(td0, null);
    			append_dev(td0, br0);
    			append_dev(tr, t0);
    			append_dev(tr, td1);

    			for (let i = 0; i < each_blocks.length; i += 1) {
    				each_blocks[i].m(td1, null);
    			}

    			append_dev(td1, br1);
    			append_dev(td1, t1);
    			append_dev(td1, button0);
    			append_dev(td1, br2);
    			append_dev(td1, t3);
    			if (if_block1) if_block1.m(td1, null);
    			insert_dev(target, t4, anchor);
    			insert_dev(target, button1, anchor);
    			append_dev(button1, t5);
    			insert_dev(target, t6, anchor);
    			if (if_block2) if_block2.m(target, anchor);
    			insert_dev(target, t7, anchor);
    			if (if_block3) if_block3.m(target, anchor);
    			insert_dev(target, t8, anchor);
    			insert_dev(target, t9, anchor);
    			insert_dev(target, t10, anchor);
    			insert_dev(target, t11, anchor);
    			insert_dev(target, t12, anchor);
    			if (if_block4) if_block4.m(target, anchor);
    			insert_dev(target, t13, anchor);
    			if (if_block5) if_block5.m(target, anchor);
    			insert_dev(target, t14, anchor);
    			insert_dev(target, button2, anchor);
    			append_dev(button2, t15);

    			if (!mounted) {
    				dispose = [
    					listen_dev(button0, "click", /*click_handler_5*/ ctx[25], false, false, false),
    					listen_dev(button1, "click", /*click_handler_6*/ ctx[26], false, false, false),
    					listen_dev(button2, "click", /*click_handler_11*/ ctx[31], false, false, false)
    				];

    				mounted = true;
    			}
    		},
    		p: function update(ctx, dirty) {
    			if (/*images*/ ctx[11][/*index*/ ctx[10]] = undefined) {
    				if (if_block0) {
    					if_block0.p(ctx, dirty);
    				} else {
    					if_block0 = create_if_block_6(ctx);
    					if_block0.c();
    					if_block0.m(td0, br0);
    				}
    			} else if (if_block0) {
    				if_block0.d(1);
    				if_block0 = null;
    			}

    			if (dirty[0] & /*seekvalue, seekvalues, set_new_label*/ 262912) {
    				each_value_3 = /*seekvalues*/ ctx[8];
    				validate_each_argument(each_value_3);
    				let i;

    				for (i = 0; i < each_value_3.length; i += 1) {
    					const child_ctx = get_each_context_3(ctx, each_value_3, i);

    					if (each_blocks[i]) {
    						each_blocks[i].p(child_ctx, dirty);
    					} else {
    						each_blocks[i] = create_each_block_3(child_ctx);
    						each_blocks[i].c();
    						each_blocks[i].m(td1, br1);
    					}
    				}

    				for (; i < each_blocks.length; i += 1) {
    					each_blocks[i].d(1);
    				}

    				each_blocks.length = each_value_3.length;
    			}

    			if (dirty[0] & /*seekvalue*/ 512) {
    				toggle_class(button0, "active", /*seekvalue*/ ctx[9] === "DELETE");
    			}

    			if (/*filter_value*/ ctx[2] === "to_check") {
    				if (if_block1) {
    					if_block1.p(ctx, dirty);
    				} else {
    					if_block1 = create_if_block_5(ctx);
    					if_block1.c();
    					if_block1.m(td1, null);
    				}
    			} else if (if_block1) {
    				if_block1.d(1);
    				if_block1 = null;
    			}

    			if (dirty[0] & /*index*/ 1024 && button1_disabled_value !== (button1_disabled_value = /*index*/ ctx[10] === 0)) {
    				prop_dev(button1, "disabled", button1_disabled_value);
    			}

    			if (/*imslen*/ ctx[0] > 100) {
    				if (if_block2) {
    					if_block2.p(ctx, dirty);
    				} else {
    					if_block2 = create_if_block_4(ctx);
    					if_block2.c();
    					if_block2.m(t7.parentNode, t7);
    				}
    			} else if (if_block2) {
    				if_block2.d(1);
    				if_block2 = null;
    			}

    			if (/*imslen*/ ctx[0] > 1000) {
    				if (if_block3) {
    					if_block3.p(ctx, dirty);
    				} else {
    					if_block3 = create_if_block_3(ctx);
    					if_block3.c();
    					if_block3.m(t8.parentNode, t8);
    				}
    			} else if (if_block3) {
    				if_block3.d(1);
    				if_block3 = null;
    			}

    			if (dirty[0] & /*index*/ 1024) set_data_dev(t9, /*index*/ ctx[10]);
    			if (dirty[0] & /*imslen*/ 1) set_data_dev(t11, /*imslen*/ ctx[0]);

    			if (/*imslen*/ ctx[0] > 1000) {
    				if (if_block4) {
    					if_block4.p(ctx, dirty);
    				} else {
    					if_block4 = create_if_block_2(ctx);
    					if_block4.c();
    					if_block4.m(t13.parentNode, t13);
    				}
    			} else if (if_block4) {
    				if_block4.d(1);
    				if_block4 = null;
    			}

    			if (/*imslen*/ ctx[0] > 100) {
    				if (if_block5) {
    					if_block5.p(ctx, dirty);
    				} else {
    					if_block5 = create_if_block_1(ctx);
    					if_block5.c();
    					if_block5.m(t14.parentNode, t14);
    				}
    			} else if (if_block5) {
    				if_block5.d(1);
    				if_block5 = null;
    			}

    			if (dirty[0] & /*index, imslen*/ 1025 && button2_disabled_value !== (button2_disabled_value = /*index*/ ctx[10] === /*imslen*/ ctx[0])) {
    				prop_dev(button2, "disabled", button2_disabled_value);
    			}
    		},
    		d: function destroy(detaching) {
    			if (detaching) detach_dev(table);
    			if (if_block0) if_block0.d();
    			destroy_each(each_blocks, detaching);
    			if (if_block1) if_block1.d();
    			if (detaching) detach_dev(t4);
    			if (detaching) detach_dev(button1);
    			if (detaching) detach_dev(t6);
    			if (if_block2) if_block2.d(detaching);
    			if (detaching) detach_dev(t7);
    			if (if_block3) if_block3.d(detaching);
    			if (detaching) detach_dev(t8);
    			if (detaching) detach_dev(t9);
    			if (detaching) detach_dev(t10);
    			if (detaching) detach_dev(t11);
    			if (detaching) detach_dev(t12);
    			if (if_block4) if_block4.d(detaching);
    			if (detaching) detach_dev(t13);
    			if (if_block5) if_block5.d(detaching);
    			if (detaching) detach_dev(t14);
    			if (detaching) detach_dev(button2);
    			mounted = false;
    			run_all(dispose);
    		}
    	};

    	dispatch_dev("SvelteRegisterBlock", {
    		block,
    		id: create_default_slot_1.name,
    		type: "slot",
    		source: "(214:0) <Box>",
    		ctx
    	});

    	return block;
    }

    // (254:4) {#each all_labels as ilabel}
    function create_each_block_2(ctx) {
    	let button;
    	let t_value = /*ilabel*/ ctx[52] + "";
    	let t;
    	let mounted;
    	let dispose;

    	function click_handler_12(...args) {
    		return /*click_handler_12*/ ctx[32](/*ilabel*/ ctx[52], ...args);
    	}

    	const block = {
    		c: function create() {
    			button = element("button");
    			t = text(t_value);
    			attr_dev(button, "class", "label svelte-1y02p24");
    			toggle_class(button, "selected", /*filter_label*/ ctx[1] === /*ilabel*/ ctx[52]);
    			add_location(button, file$1, 254, 8, 7408);
    		},
    		m: function mount(target, anchor) {
    			insert_dev(target, button, anchor);
    			append_dev(button, t);

    			if (!mounted) {
    				dispose = listen_dev(button, "click", click_handler_12, false, false, false);
    				mounted = true;
    			}
    		},
    		p: function update(new_ctx, dirty) {
    			ctx = new_ctx;
    			if (dirty[0] & /*all_labels*/ 4096 && t_value !== (t_value = /*ilabel*/ ctx[52] + "")) set_data_dev(t, t_value);

    			if (dirty[0] & /*filter_label, all_labels*/ 4098) {
    				toggle_class(button, "selected", /*filter_label*/ ctx[1] === /*ilabel*/ ctx[52]);
    			}
    		},
    		d: function destroy(detaching) {
    			if (detaching) detach_dev(button);
    			mounted = false;
    			dispose();
    		}
    	};

    	dispatch_dev("SvelteRegisterBlock", {
    		block,
    		id: create_each_block_2.name,
    		type: "each",
    		source: "(254:4) {#each all_labels as ilabel}",
    		ctx
    	});

    	return block;
    }

    // (262:4) {#if filter_label !== 'none'}
    function create_if_block(ctx) {
    	let each_1_anchor;
    	let each_value_1 = /*filter_values*/ ctx[3];
    	validate_each_argument(each_value_1);
    	let each_blocks = [];

    	for (let i = 0; i < each_value_1.length; i += 1) {
    		each_blocks[i] = create_each_block_1(get_each_context_1(ctx, each_value_1, i));
    	}

    	const block = {
    		c: function create() {
    			for (let i = 0; i < each_blocks.length; i += 1) {
    				each_blocks[i].c();
    			}

    			each_1_anchor = empty();
    		},
    		m: function mount(target, anchor) {
    			for (let i = 0; i < each_blocks.length; i += 1) {
    				each_blocks[i].m(target, anchor);
    			}

    			insert_dev(target, each_1_anchor, anchor);
    		},
    		p: function update(ctx, dirty) {
    			if (dirty[0] & /*filter_value, filter_values, loadData*/ 131084) {
    				each_value_1 = /*filter_values*/ ctx[3];
    				validate_each_argument(each_value_1);
    				let i;

    				for (i = 0; i < each_value_1.length; i += 1) {
    					const child_ctx = get_each_context_1(ctx, each_value_1, i);

    					if (each_blocks[i]) {
    						each_blocks[i].p(child_ctx, dirty);
    					} else {
    						each_blocks[i] = create_each_block_1(child_ctx);
    						each_blocks[i].c();
    						each_blocks[i].m(each_1_anchor.parentNode, each_1_anchor);
    					}
    				}

    				for (; i < each_blocks.length; i += 1) {
    					each_blocks[i].d(1);
    				}

    				each_blocks.length = each_value_1.length;
    			}
    		},
    		d: function destroy(detaching) {
    			destroy_each(each_blocks, detaching);
    			if (detaching) detach_dev(each_1_anchor);
    		}
    	};

    	dispatch_dev("SvelteRegisterBlock", {
    		block,
    		id: create_if_block.name,
    		type: "if",
    		source: "(262:4) {#if filter_label !== 'none'}",
    		ctx
    	});

    	return block;
    }

    // (263:4) {#each filter_values as lv}
    function create_each_block_1(ctx) {
    	let button;
    	let t_value = /*lv*/ ctx[47] + "";
    	let t;
    	let mounted;
    	let dispose;

    	function click_handler_14(...args) {
    		return /*click_handler_14*/ ctx[34](/*lv*/ ctx[47], ...args);
    	}

    	const block = {
    		c: function create() {
    			button = element("button");
    			t = text(t_value);
    			attr_dev(button, "class", "svelte-1y02p24");
    			toggle_class(button, "selected", /*filter_value*/ ctx[2] === /*lv*/ ctx[47]);
    			add_location(button, file$1, 263, 8, 7811);
    		},
    		m: function mount(target, anchor) {
    			insert_dev(target, button, anchor);
    			append_dev(button, t);

    			if (!mounted) {
    				dispose = listen_dev(button, "click", click_handler_14, false, false, false);
    				mounted = true;
    			}
    		},
    		p: function update(new_ctx, dirty) {
    			ctx = new_ctx;
    			if (dirty[0] & /*filter_values*/ 8 && t_value !== (t_value = /*lv*/ ctx[47] + "")) set_data_dev(t, t_value);

    			if (dirty[0] & /*filter_value, filter_values*/ 12) {
    				toggle_class(button, "selected", /*filter_value*/ ctx[2] === /*lv*/ ctx[47]);
    			}
    		},
    		d: function destroy(detaching) {
    			if (detaching) detach_dev(button);
    			mounted = false;
    			dispose();
    		}
    	};

    	dispatch_dev("SvelteRegisterBlock", {
    		block,
    		id: create_each_block_1.name,
    		type: "each",
    		source: "(263:4) {#each filter_values as lv}",
    		ctx
    	});

    	return block;
    }

    // (267:4) {#each ['up', 'low', 'none'] as lv}
    function create_each_block(ctx) {
    	let button;
    	let t;
    	let mounted;
    	let dispose;

    	function click_handler_15(...args) {
    		return /*click_handler_15*/ ctx[35](/*lv*/ ctx[47], ...args);
    	}

    	const block = {
    		c: function create() {
    			button = element("button");
    			t = text(/*lv*/ ctx[47]);
    			attr_dev(button, "class", "svelte-1y02p24");
    			toggle_class(button, "selected", /*filter_size*/ ctx[4] === /*lv*/ ctx[47]);
    			add_location(button, file$1, 267, 8, 7993);
    		},
    		m: function mount(target, anchor) {
    			insert_dev(target, button, anchor);
    			append_dev(button, t);

    			if (!mounted) {
    				dispose = listen_dev(button, "click", click_handler_15, false, false, false);
    				mounted = true;
    			}
    		},
    		p: function update(new_ctx, dirty) {
    			ctx = new_ctx;

    			if (dirty[0] & /*filter_size*/ 16) {
    				toggle_class(button, "selected", /*filter_size*/ ctx[4] === /*lv*/ ctx[47]);
    			}
    		},
    		d: function destroy(detaching) {
    			if (detaching) detach_dev(button);
    			mounted = false;
    			dispose();
    		}
    	};

    	dispatch_dev("SvelteRegisterBlock", {
    		block,
    		id: create_each_block.name,
    		type: "each",
    		source: "(267:4) {#each ['up', 'low', 'none'] as lv}",
    		ctx
    	});

    	return block;
    }

    // (252:0) <Box>
    function create_default_slot(ctx) {
    	let h3;
    	let t0;
    	let t1_value = /*images*/ ctx[11].length + "";
    	let t1;
    	let t2;
    	let t3;
    	let t4;
    	let button;
    	let t6;
    	let br;
    	let t7;
    	let t8;
    	let each1_anchor;
    	let mounted;
    	let dispose;
    	let each_value_2 = /*all_labels*/ ctx[12];
    	validate_each_argument(each_value_2);
    	let each_blocks_1 = [];

    	for (let i = 0; i < each_value_2.length; i += 1) {
    		each_blocks_1[i] = create_each_block_2(get_each_context_2(ctx, each_value_2, i));
    	}

    	let if_block = /*filter_label*/ ctx[1] !== "none" && create_if_block(ctx);
    	let each_value = ["up", "low", "none"];
    	validate_each_argument(each_value);
    	let each_blocks = [];

    	for (let i = 0; i < 3; i += 1) {
    		each_blocks[i] = create_each_block(get_each_context(ctx, each_value, i));
    	}

    	const block = {
    		c: function create() {
    			h3 = element("h3");
    			t0 = text("select labels filter: total ");
    			t1 = text(t1_value);
    			t2 = text(" imgs");
    			t3 = space();

    			for (let i = 0; i < each_blocks_1.length; i += 1) {
    				each_blocks_1[i].c();
    			}

    			t4 = space();
    			button = element("button");
    			button.textContent = "All";
    			t6 = space();
    			br = element("br");
    			t7 = space();
    			if (if_block) if_block.c();
    			t8 = space();

    			for (let i = 0; i < 3; i += 1) {
    				each_blocks[i].c();
    			}

    			each1_anchor = empty();
    			add_location(h3, file$1, 252, 4, 7309);
    			attr_dev(button, "class", "label svelte-1y02p24");
    			toggle_class(button, "selected", /*filter_label*/ ctx[1] === "none");
    			add_location(button, file$1, 257, 4, 7578);
    			add_location(br, file$1, 260, 4, 7732);
    		},
    		m: function mount(target, anchor) {
    			insert_dev(target, h3, anchor);
    			append_dev(h3, t0);
    			append_dev(h3, t1);
    			append_dev(h3, t2);
    			insert_dev(target, t3, anchor);

    			for (let i = 0; i < each_blocks_1.length; i += 1) {
    				each_blocks_1[i].m(target, anchor);
    			}

    			insert_dev(target, t4, anchor);
    			insert_dev(target, button, anchor);
    			insert_dev(target, t6, anchor);
    			insert_dev(target, br, anchor);
    			insert_dev(target, t7, anchor);
    			if (if_block) if_block.m(target, anchor);
    			insert_dev(target, t8, anchor);

    			for (let i = 0; i < 3; i += 1) {
    				each_blocks[i].m(target, anchor);
    			}

    			insert_dev(target, each1_anchor, anchor);

    			if (!mounted) {
    				dispose = listen_dev(button, "click", /*click_handler_13*/ ctx[33], false, false, false);
    				mounted = true;
    			}
    		},
    		p: function update(ctx, dirty) {
    			if (dirty[0] & /*images*/ 2048 && t1_value !== (t1_value = /*images*/ ctx[11].length + "")) set_data_dev(t1, t1_value);

    			if (dirty[0] & /*filter_label, all_labels, loadData*/ 135170) {
    				each_value_2 = /*all_labels*/ ctx[12];
    				validate_each_argument(each_value_2);
    				let i;

    				for (i = 0; i < each_value_2.length; i += 1) {
    					const child_ctx = get_each_context_2(ctx, each_value_2, i);

    					if (each_blocks_1[i]) {
    						each_blocks_1[i].p(child_ctx, dirty);
    					} else {
    						each_blocks_1[i] = create_each_block_2(child_ctx);
    						each_blocks_1[i].c();
    						each_blocks_1[i].m(t4.parentNode, t4);
    					}
    				}

    				for (; i < each_blocks_1.length; i += 1) {
    					each_blocks_1[i].d(1);
    				}

    				each_blocks_1.length = each_value_2.length;
    			}

    			if (dirty[0] & /*filter_label*/ 2) {
    				toggle_class(button, "selected", /*filter_label*/ ctx[1] === "none");
    			}

    			if (/*filter_label*/ ctx[1] !== "none") {
    				if (if_block) {
    					if_block.p(ctx, dirty);
    				} else {
    					if_block = create_if_block(ctx);
    					if_block.c();
    					if_block.m(t8.parentNode, t8);
    				}
    			} else if (if_block) {
    				if_block.d(1);
    				if_block = null;
    			}

    			if (dirty[0] & /*filter_size, loadData*/ 131088) {
    				each_value = ["up", "low", "none"];
    				validate_each_argument(each_value);
    				let i;

    				for (i = 0; i < 3; i += 1) {
    					const child_ctx = get_each_context(ctx, each_value, i);

    					if (each_blocks[i]) {
    						each_blocks[i].p(child_ctx, dirty);
    					} else {
    						each_blocks[i] = create_each_block(child_ctx);
    						each_blocks[i].c();
    						each_blocks[i].m(each1_anchor.parentNode, each1_anchor);
    					}
    				}

    				for (; i < 3; i += 1) {
    					each_blocks[i].d(1);
    				}
    			}
    		},
    		d: function destroy(detaching) {
    			if (detaching) detach_dev(h3);
    			if (detaching) detach_dev(t3);
    			destroy_each(each_blocks_1, detaching);
    			if (detaching) detach_dev(t4);
    			if (detaching) detach_dev(button);
    			if (detaching) detach_dev(t6);
    			if (detaching) detach_dev(br);
    			if (detaching) detach_dev(t7);
    			if (if_block) if_block.d(detaching);
    			if (detaching) detach_dev(t8);
    			destroy_each(each_blocks, detaching);
    			if (detaching) detach_dev(each1_anchor);
    			mounted = false;
    			dispose();
    		}
    	};

    	dispatch_dev("SvelteRegisterBlock", {
    		block,
    		id: create_default_slot.name,
    		type: "slot",
    		source: "(252:0) <Box>",
    		ctx
    	});

    	return block;
    }

    function create_fragment$1(ctx) {
    	let box0;
    	let t0;
    	let box1;
    	let t1;
    	let box2;
    	let current;
    	let mounted;
    	let dispose;

    	box0 = new Box({
    			props: {
    				$$slots: { default: [create_default_slot_2] },
    				$$scope: { ctx }
    			},
    			$$inline: true
    		});

    	box1 = new Box({
    			props: {
    				$$slots: { default: [create_default_slot_1] },
    				$$scope: { ctx }
    			},
    			$$inline: true
    		});

    	box2 = new Box({
    			props: {
    				$$slots: { default: [create_default_slot] },
    				$$scope: { ctx }
    			},
    			$$inline: true
    		});

    	const block = {
    		c: function create() {
    			create_component(box0.$$.fragment);
    			t0 = space();
    			create_component(box1.$$.fragment);
    			t1 = space();
    			create_component(box2.$$.fragment);
    		},
    		l: function claim(nodes) {
    			throw new Error("options.hydrate only works if the component was compiled with the `hydratable: true` option");
    		},
    		m: function mount(target, anchor) {
    			mount_component(box0, target, anchor);
    			insert_dev(target, t0, anchor);
    			mount_component(box1, target, anchor);
    			insert_dev(target, t1, anchor);
    			mount_component(box2, target, anchor);
    			current = true;

    			if (!mounted) {
    				dispose = listen_dev(window, "keydown", /*handleKeydown*/ ctx[16], false, false, false);
    				mounted = true;
    			}
    		},
    		p: function update(ctx, dirty) {
    			const box0_changes = {};

    			if (dirty[0] & /*seek_only_clear, all_labels, seek_label, storetext*/ 12480 | dirty[1] & /*$$scope*/ 536870912) {
    				box0_changes.$$scope = { dirty, ctx };
    			}

    			box0.$set(box0_changes);
    			const box1_changes = {};

    			if (dirty[0] & /*index, imslen, filter_textes, filter_value, seekvalue, seekvalues, images*/ 3877 | dirty[1] & /*$$scope*/ 536870912) {
    				box1_changes.$$scope = { dirty, ctx };
    			}

    			box1.$set(box1_changes);
    			const box2_changes = {};

    			if (dirty[0] & /*filter_size, filter_values, filter_value, filter_label, all_labels, images*/ 6174 | dirty[1] & /*$$scope*/ 536870912) {
    				box2_changes.$$scope = { dirty, ctx };
    			}

    			box2.$set(box2_changes);
    		},
    		i: function intro(local) {
    			if (current) return;
    			transition_in(box0.$$.fragment, local);
    			transition_in(box1.$$.fragment, local);
    			transition_in(box2.$$.fragment, local);
    			current = true;
    		},
    		o: function outro(local) {
    			transition_out(box0.$$.fragment, local);
    			transition_out(box1.$$.fragment, local);
    			transition_out(box2.$$.fragment, local);
    			current = false;
    		},
    		d: function destroy(detaching) {
    			destroy_component(box0, detaching);
    			if (detaching) detach_dev(t0);
    			destroy_component(box1, detaching);
    			if (detaching) detach_dev(t1);
    			destroy_component(box2, detaching);
    			mounted = false;
    			dispose();
    		}
    	};

    	dispatch_dev("SvelteRegisterBlock", {
    		block,
    		id: create_fragment$1.name,
    		type: "component",
    		source: "",
    		ctx
    	});

    	return block;
    }

    function instance$1($$self, $$props, $$invalidate) {
    	let source_server = "http://0.0.0.0:8081";
    	let ims = {}; // ret dict
    	let imslen = 0;
    	let filter_label = "none"; //
    	let filter_value = "all";
    	let filter_values = [];
    	let filter_size = "none"; // up low
    	let filter_textes = [];
    	let counts_table = "";
    	let seek_label = "none";
    	let seek_only_clear = "no";
    	let seekvalues = [];
    	let seekvalue = "";
    	let seekrec = {};
    	let seek_resp = {};
    	let index = 0;
    	let images = [];
    	let all_labels = [];
    	let label_data = {};
    	let key;
    	let keyCode;
    	let nums = [96, 97, 98, 99, 100, 101, 102, 103, 104, 105];
    	let nums2 = [48, 49, 50, 51, 52, 53, 54, 55, 56, 57];
    	let storetext = "store:";
    	let store_rec = {};

    	async function read_label() {
    		const res = await fetch(`${source_server}/get_label_value_on_image/${seek_label}/${images[index]}`);
    		seekrec = await res.json();
    		$$invalidate(9, seekvalue = seekrec.imlabel);
    		console.log("seelvalue=", seekvalue);
    	}

    	function on_seek() {
    		console.log("seek");
    		read_label();
    	}

    	function handleKeydown(event) {
    		key = event.key;
    		keyCode = event.keyCode;
    		console.log("key=", key, " keycode=", keyCode);

    		if (keyCode === 37 && index > 0) {
    			$$invalidate(10, index--, index);
    			on_seek();
    		}

    		if (keyCode === 39 && index < imslen) {
    			$$invalidate(10, index++, index);
    			on_seek();
    		}

    		if (keyCode === 32 && index < imslen) {
    			$$invalidate(10, index++, index);
    			on_seek();
    		}

    		if (nums.includes(keyCode) && keyCode - 96 < seekvalues.length) {
    			$$invalidate(9, seekvalue = seekvalues[keyCode - 96]);
    			set_new_label();
    		}

    		if (nums2.includes(keyCode) && keyCode - 48 < seekvalues.length) {
    			$$invalidate(9, seekvalue = seekvalues[keyCode - 48]);
    			set_new_label();
    		}

    		if (keyCode === 83 && seek_label !== "none") {
    			seek_store();
    		}
    	}

    	async function loadData() {
    		console.log("in", filter_label, filter_value, seek_label);
    		const res = await fetch(`${source_server}/set_filter/${filter_label}/${filter_value}/${seek_label}/${seek_only_clear}/${filter_size}`);
    		ims = await res.json();
    		console.log("ims", ims);
    		$$invalidate(11, images = ims.images);
    		$$invalidate(0, imslen = images.length);
    		$$invalidate(10, index = 0);
    		$$invalidate(12, all_labels = ims.labels);
    		label_data = ims.label;
    		$$invalidate(3, filter_values = ims.values);
    		filter_values.push("all");
    		filter_values.push("to_check");
    		$$invalidate(5, filter_textes = ims.text);
    		console.log("filtertextes =", filter_textes);
    		$$invalidate(8, seekvalues = ims.seekvalues);
    		counts_table = ims.counts;
    		console.log("labelvalues =", filter_values);
    		console.log("set filter");
    		read_label();
    	}

    	onMount(async () => {
    		await loadData();
    	});

    	async function set_new_label() {
    		const res = await fetch(`${source_server}/set_value/${images[index]}/${seek_label}/${seekvalue}`);
    		seek_resp = await res.json();

    		if (seek_resp.res !== "ok") {
    			alert(seek_resp.res);
    		}

    		console.log("set_new_label=", seek_resp);
    		console.log(seek_resp);
    	}

    	async function seek_store() {
    		console.log(`storetext ${source_server}/store_label/${seek_label}`);
    		const res = await fetch(`${source_server}/store_label/${seek_label}`);
    		store_rec = await res.json();
    		$$invalidate(13, storetext = store_rec.res);
    		console.log("storetext:", storetext, "label:", seek_label);

    		setTimeout(
    			function () {
    				$$invalidate(13, storetext = "store");
    				console.log("storetext finish");
    			},
    			3000
    		);
    	}

    	console.log("started");
    	const writable_props = [];

    	Object.keys($$props).forEach(key => {
    		if (!~writable_props.indexOf(key) && key.slice(0, 2) !== "$$") console_1.warn(`<App> was created with unknown prop '${key}'`);
    	});

    	let { $$slots = {}, $$scope } = $$props;
    	validate_slots("App", $$slots, []);

    	const click_handler = () => {
    		seek_store();
    	};

    	const click_handler_1 = ilabel => {
    		$$invalidate(6, seek_label = ilabel);
    		loadData();
    	};

    	const click_handler_2 = () => {
    		$$invalidate(7, seek_only_clear = "yes");
    		loadData();
    	};

    	const click_handler_3 = () => {
    		$$invalidate(7, seek_only_clear = "no");
    		loadData();
    	};

    	const click_handler_4 = sv => {
    		$$invalidate(9, seekvalue = sv);
    		set_new_label();
    	};

    	const click_handler_5 = () => {
    		$$invalidate(9, seekvalue = "DELETE");
    		set_new_label();
    	};

    	const click_handler_6 = () => {
    		$$invalidate(10, index--, index);
    		on_seek();
    	};

    	const click_handler_7 = () => {
    		$$invalidate(10, index -= 100);
    		on_seek();
    	};

    	const click_handler_8 = () => {
    		$$invalidate(10, index -= 1000);
    		on_seek();
    	};

    	const click_handler_9 = () => {
    		$$invalidate(10, index += 1000);
    		on_seek();
    	};

    	const click_handler_10 = () => {
    		$$invalidate(10, index += 100);
    		on_seek();
    	};

    	const click_handler_11 = () => {
    		$$invalidate(10, index++, index);
    		on_seek();
    	};

    	const click_handler_12 = ilabel => {
    		$$invalidate(1, filter_label = ilabel);
    		loadData();
    	};

    	const click_handler_13 = () => {
    		$$invalidate(1, filter_label = "none");
    		loadData();
    	};

    	const click_handler_14 = lv => {
    		$$invalidate(2, filter_value = lv);
    		loadData();
    	};

    	const click_handler_15 = lv => {
    		$$invalidate(4, filter_size = lv);
    		loadData();
    	};

    	$$self.$capture_state = () => ({
    		onMount,
    		Box,
    		source_server,
    		ims,
    		imslen,
    		filter_label,
    		filter_value,
    		filter_values,
    		filter_size,
    		filter_textes,
    		counts_table,
    		seek_label,
    		seek_only_clear,
    		seekvalues,
    		seekvalue,
    		seekrec,
    		seek_resp,
    		index,
    		images,
    		all_labels,
    		label_data,
    		key,
    		keyCode,
    		nums,
    		nums2,
    		storetext,
    		store_rec,
    		read_label,
    		on_seek,
    		handleKeydown,
    		loadData,
    		set_new_label,
    		seek_store
    	});

    	$$self.$inject_state = $$props => {
    		if ("source_server" in $$props) $$invalidate(14, source_server = $$props.source_server);
    		if ("ims" in $$props) ims = $$props.ims;
    		if ("imslen" in $$props) $$invalidate(0, imslen = $$props.imslen);
    		if ("filter_label" in $$props) $$invalidate(1, filter_label = $$props.filter_label);
    		if ("filter_value" in $$props) $$invalidate(2, filter_value = $$props.filter_value);
    		if ("filter_values" in $$props) $$invalidate(3, filter_values = $$props.filter_values);
    		if ("filter_size" in $$props) $$invalidate(4, filter_size = $$props.filter_size);
    		if ("filter_textes" in $$props) $$invalidate(5, filter_textes = $$props.filter_textes);
    		if ("counts_table" in $$props) counts_table = $$props.counts_table;
    		if ("seek_label" in $$props) $$invalidate(6, seek_label = $$props.seek_label);
    		if ("seek_only_clear" in $$props) $$invalidate(7, seek_only_clear = $$props.seek_only_clear);
    		if ("seekvalues" in $$props) $$invalidate(8, seekvalues = $$props.seekvalues);
    		if ("seekvalue" in $$props) $$invalidate(9, seekvalue = $$props.seekvalue);
    		if ("seekrec" in $$props) seekrec = $$props.seekrec;
    		if ("seek_resp" in $$props) seek_resp = $$props.seek_resp;
    		if ("index" in $$props) $$invalidate(10, index = $$props.index);
    		if ("images" in $$props) $$invalidate(11, images = $$props.images);
    		if ("all_labels" in $$props) $$invalidate(12, all_labels = $$props.all_labels);
    		if ("label_data" in $$props) label_data = $$props.label_data;
    		if ("key" in $$props) key = $$props.key;
    		if ("keyCode" in $$props) keyCode = $$props.keyCode;
    		if ("nums" in $$props) nums = $$props.nums;
    		if ("nums2" in $$props) nums2 = $$props.nums2;
    		if ("storetext" in $$props) $$invalidate(13, storetext = $$props.storetext);
    		if ("store_rec" in $$props) store_rec = $$props.store_rec;
    	};

    	if ($$props && "$$inject" in $$props) {
    		$$self.$inject_state($$props.$$inject);
    	}

    	return [
    		imslen,
    		filter_label,
    		filter_value,
    		filter_values,
    		filter_size,
    		filter_textes,
    		seek_label,
    		seek_only_clear,
    		seekvalues,
    		seekvalue,
    		index,
    		images,
    		all_labels,
    		storetext,
    		source_server,
    		on_seek,
    		handleKeydown,
    		loadData,
    		set_new_label,
    		seek_store,
    		click_handler,
    		click_handler_1,
    		click_handler_2,
    		click_handler_3,
    		click_handler_4,
    		click_handler_5,
    		click_handler_6,
    		click_handler_7,
    		click_handler_8,
    		click_handler_9,
    		click_handler_10,
    		click_handler_11,
    		click_handler_12,
    		click_handler_13,
    		click_handler_14,
    		click_handler_15
    	];
    }

    class App extends SvelteComponentDev {
    	constructor(options) {
    		super(options);
    		init(this, options, instance$1, create_fragment$1, safe_not_equal, {}, [-1, -1]);

    		dispatch_dev("SvelteRegisterComponent", {
    			component: this,
    			tagName: "App",
    			options,
    			id: create_fragment$1.name
    		});
    	}
    }

    var app = new App({
    	target: document.body
    });

    return app;

}());
//# sourceMappingURL=bundle.js.map

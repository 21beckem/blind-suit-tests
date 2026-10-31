/*!
 * vr.js - phone-VR display for low-resolution depth / haptic simulation.
 * Load with <script src="vr.js"></script>. Exposes a ready-made global `VR`
 * (an instance of `VRDisplay`).
 *
 *   VR.patchConfig(key, value)       change one setting (saved, UI updated)
 *   VR.configPanelsNode              node holding every setting panel (mount it anywhere)
 *   VR.enableConfigPopup(bool)       bottom-right button that opens the panels in a popup
 *   VR.previewNode                   node showing the live image (mount it anywhere)
 *   VR.enter() / VR.leave()          fullscreen VR mode (double-tap also leaves)
 *   VR.paint(grid)                   grid = 2D array [row][col] of 0..1 (null/NaN = no reading)
 *   VR.startStreamingDemoData() / VR.stopStreamingDemoData()
 *
 * Extras: VR.config (copy of current settings), VR.demoRunning, VR.isInVR.
 */
(function () {
    'use strict';

    const KEY = 'vrjs.config.v1';
    const DEFAULTS = {
        levels: 0, res: 0, blur: 0, rate: 60, lag: 0, tau: 0, noise: 0,
        size: 0.7, shift: 0.06, voff: 0, k1: 0.01, k2: 0.01, gap: 0.06,
        smooth: false, flipH: false, flipV: false, swap: false, test: false,
        rotate: true, rotFlip: false
    };
    // [key, label, min, max, step]
    const SIM = [
        ['levels', 'Intensity levels (0 = continuous)', 0, 16, 1],
        ['res', 'Grid resolution (cells on longest side, 0 = native)', 0, 32, 1],
        ['blur', 'Spread between cells (radius)', 0, 3, 1],
        ['rate', 'Update rate (Hz, 60 = max)', 1, 60, 1],
        ['lag', 'Lag (ms)', 0, 1000, 10],
        ['tau', 'Adaptation fade (s, 0 = off)', 0, 10, 0.5],
        ['noise', 'Noise', 0, 0.5, 0.01]
    ];
    const LAY = [
        ['size', 'Image size', 0.2, 1.2, 0.01],
        ['shift', 'Eye spacing shift', -0.15, 0.15, 0.005],
        ['voff', 'Vertical offset', -0.3, 0.3, 0.005],
        ['k1', 'Lens warp k1', -0.5, 1, 0.01],
        ['k2', 'Lens warp k2', -0.5, 1, 0.01],
        ['gap', 'Cell gap', 0, 0.3, 0.01]
    ];
    const RANGE = {};
    for (const [k, , mn, mx] of SIM.concat(LAY)) RANGE[k] = [mn, mx];

    const CSS = `
.vrjs-panels{font:14px/1.4 system-ui,sans-serif;color:#1d2330;background:#fff;box-sizing:border-box;max-width:100%}
.vrjs-panels *{box-sizing:border-box}
.vrjs-panels details{border:1px solid #d8dde6;border-radius:8px;margin:8px 0;padding:4px 12px;background:#f7f9fc}
.vrjs-panels summary{cursor:pointer;padding:6px 0;font-weight:600}
.vrjs-ctl{display:grid;grid-template-columns:1fr 64px;gap:2px 8px;margin:8px 0}
.vrjs-ctl label{grid-column:1/3;font-size:13px;color:#5a6577}
.vrjs-ctl input[type=range]{width:100%;accent-color:#1f7a5c}
.vrjs-ctl output{text-align:right;font-variant-numeric:tabular-nums}
.vrjs-chk{display:flex;gap:8px;align-items:center;margin:8px 0}
.vrjs-btn{font:inherit;font-size:14px;background:#fff;color:#1d2330;border:1px solid #c5ccd8;border-radius:8px;padding:8px 14px;cursor:pointer}
.vrjs-btn:focus-visible,.vrjs-panels input:focus-visible{outline:2px solid #1f7a5c;outline-offset:2px}
.vrjs-preview{width:100%;aspect-ratio:2/1;background:#000;border-radius:8px}
.vrjs-preview.vrjs-tmp{position:fixed;left:0;top:0;width:0;height:0;aspect-ratio:auto}
.vrjs-canvas{display:block;width:100%;height:100%;border-radius:8px;background:#000}
.vrjs-canvas.vrjs-full{position:fixed;inset:0;width:100vw;height:100vh;border-radius:0;z-index:2147483000}
body.vrjs-vr{overflow:hidden}
.vrjs-fab{position:fixed;right:14px;bottom:calc(14px + env(safe-area-inset-bottom,0px));z-index:2147482000;box-shadow:0 2px 10px rgba(0,0,0,.25)}
body.vrjs-vr .vrjs-fab{display:none}
.vrjs-pop{position:fixed;inset:0;z-index:2147482500;background:rgba(20,25,35,.45);display:flex;align-items:flex-end;justify-content:center;font-family:system-ui,sans-serif}
.vrjs-card{background:#fff;color:#1d2330;width:min(560px,100%);max-height:85vh;overflow:auto;border-radius:14px 14px 0 0;padding:12px 16px calc(16px + env(safe-area-inset-bottom,0px))}
.vrjs-head{display:flex;justify-content:space-between;align-items:center;font-weight:600;font-size:16px}
`;

    const VS = 'attribute vec2 a;void main(){gl_Position=vec4(a,0.,1.);}';
    const FS = `precision mediump float;
uniform vec2 res,dims;uniform sampler2D tex;uniform float aspect,k1,k2,size,shift,voff,gap;
void main(){vec2 f=gl_FragCoord.xy;float hw=res.x*.5;bool R=f.x>hw;float ex=f.x-(R?hw:0.);
vec2 c=vec2(hw*.5+(R?-1.:1.)*shift*res.x,res.y*.5+voff*res.y);float hs=res.y*size*.5;
vec2 p=(vec2(ex,f.y)-c)/hs;float r2=dot(p,p);vec2 q=p*(1.+k1*r2+k2*r2*r2);
vec2 uv=vec2(q.x/(2.*aspect)+.5,.5-q.y*.5);float v=0.;
if(uv.x>=0.&&uv.x<=1.&&uv.y>=0.&&uv.y<=1.){v=texture2D(tex,uv).r;vec2 g=abs(fract(uv*dims)-.5)*2.;v*=1.-step(1.-gap,max(g.x,g.y));}
gl_FragColor=vec4(vec3(v),1.);}`;

    class VRDisplay {
        constructor() {
            this.cfg = Object.assign({}, DEFAULTS);
            try { Object.assign(this.cfg, JSON.parse(localStorage.getItem(KEY) || '{}')); } catch (e) { }
            for (const k of Object.keys(DEFAULTS)) this.cfg[k] = this._coerce(k, this.cfg[k]);

            this._hist = [];
            this._img = { w: 8, h: 8, b: new Uint8Array(64) };
            this._ad = new Float32Array(0);
            this._lastUp = 0;
            this._dirty = true;
            this._ui = {};
            this._demo = null;
            this._vr = false;
            this._wake = null;
            this._lastTap = 0;
            this._dir = 1;
            this._onMotion = null;
            this._panels = null;
            this._popupOn = false;

            this._injectStyle();
            this._buildPreview();
            this._initGL();

            document.addEventListener('fullscreenchange', () => {
                if (!document.fullscreenElement && this._vr) this.leave();
            });
            window.addEventListener('resize', () => this._resize());
            if (window.ResizeObserver) new ResizeObserver(() => this._resize()).observe(this._canvas);
            requestAnimationFrame(t => this._loop(t));

            this._startMotion();
        }

        /* ---------------- public API ---------------- */
        get config() { return Object.assign({}, this.cfg); }
        get demoRunning() { return !!this._demo; }
        get isInVR() { return this._vr; }
        get previewNode() { return this._wrap; }
        get configPanelsNode() { if (!this._panels) this._buildPanels(); return this._panels; }

        patchConfig(key, value) {
            if (!(key in DEFAULTS)) throw new Error('VR.patchConfig: unknown key "' + key + '"');
            this.cfg[key] = this._coerce(key, value);
            this._saveCfg();
            if (this._ui[key]) this._ui[key](this.cfg[key]);
            this._dirty = true;
            if (this._vr && (key === 'rotate' || key === 'rotFlip')) { this._applyRot(); this._resize(); }
        }

        paint(grid) {
            this.stopStreamingDemoData();
            this._push(grid);
        }

        startStreamingDemoData() {
            if (this._demo) return;
            let t = 0;
            this._demo = setInterval(() => {
                t += 0.07;
                const bx = 3.5 + 3 * Math.sin(t), by = 3.5 + 2 * Math.cos(t * 0.7), br = 0.5 + 0.5 * Math.sin(t * 1.3);
                const g = [];
                for (let y = 0; y < 8; y++) {
                    const row = [];
                    for (let x = 0; x < 8; x++) {
                        let v = 0.12 + y * 0.01;
                        if (x < 2) v = 0.45;
                        if (Math.hypot(x - bx, y - by) < 1.9) v = 0.6 + 0.4 * br;
                        row.push(v);
                    }
                    g.push(row);
                }
                this._push(g);
            }, 66);
        }
        stopStreamingDemoData() {
            if (this._demo) { clearInterval(this._demo); this._demo = null; }
        }

        enableConfigPopup(on) {
            this._popupOn = !!on;
            const run = () => {
                if (this._popupOn) {
                    if (!this._fab) {
                        this._fab = document.createElement('button');
                        this._fab.className = 'vrjs-btn vrjs-fab';
                        this._fab.type = 'button';
                        this._fab.textContent = '⚙ Display settings';
                        this._fab.onclick = () => this._openPopup();
                    }
                    document.body.appendChild(this._fab);
                } else {
                    if (this._fab) this._fab.remove();
                    this._closePopup();
                }
            };
            if (document.body) run(); else document.addEventListener('DOMContentLoaded', run, { once: true });
        }

        async enter() {
            if (this._vr) return;
            this._vr = true;
            // iOS only: motion permission must be requested inside the tap, before any await
            let perm = null;
            if (window.DeviceMotionEvent && typeof DeviceMotionEvent.requestPermission === 'function') {
                perm = DeviceMotionEvent.requestPermission().catch(() => { });
            }
            this._closePopup();
            if (!this._wrap.isConnected) {
                this._wrap.classList.add('vrjs-tmp');
                document.body.appendChild(this._wrap);
                this._tmpMounted = true;
            }
            document.body.classList.add('vrjs-vr');
            this._canvas.classList.add('vrjs-full');
            this._applyRot();
            if (perm) await perm;
            try { await document.documentElement.requestFullscreen(); } catch (e) { }
            if (!this.cfg.rotate) { try { await screen.orientation.lock('landscape'); } catch (e) { } }
            try { this._wake = await navigator.wakeLock.request('screen'); } catch (e) { }
            setTimeout(() => this._resize(), 150);
        }

        leave() {
            if (!this._vr) return;
            this._vr = false;
            document.body.classList.remove('vrjs-vr');
            this._canvas.classList.remove('vrjs-full');
            //   this._stopMotion();
            this._applyRot();
            if (document.fullscreenElement) document.exitFullscreen().catch(() => { });
            try { screen.orientation.unlock(); } catch (e) { }
            if (this._wake) { this._wake.release().catch(() => { }); this._wake = null; }
            if (this._tmpMounted) { this._wrap.remove(); this._wrap.classList.remove('vrjs-tmp'); this._tmpMounted = false; }
            setTimeout(() => this._resize(), 150);
        }

        /* ---------------- internals: setup ---------------- */
        _coerce(k, v) {
            if (typeof DEFAULTS[k] === 'boolean') return !!v;
            const n = Number(v);
            if (!isFinite(n)) return DEFAULTS[k];
            const [mn, mx] = RANGE[k];
            return Math.min(mx, Math.max(mn, n));
        }
        _saveCfg() { try { localStorage.setItem(KEY, JSON.stringify(this.cfg)); } catch (e) { } }

        _injectStyle() {
            if (document.getElementById('vrjs-style')) return;
            const s = document.createElement('style');
            s.id = 'vrjs-style';
            s.textContent = CSS;
            (document.head || document.documentElement).appendChild(s);
        }

        _buildPreview() {
            this._wrap = document.createElement('div');
            this._wrap.className = 'vrjs-preview';
            this._canvas = document.createElement('canvas');
            this._canvas.className = 'vrjs-canvas';
            this._wrap.appendChild(this._canvas);
            this._canvas.addEventListener('pointerup', () => {
                const n = Date.now();
                if (this._vr && n - this._lastTap < 400) this.leave();
                this._lastTap = n;
            });
        }

        _buildPanels() {
            const root = document.createElement('div');
            root.className = 'vrjs-panels';
            const slider = (host, [k, lab, mn, mx, st]) => {
                const d = document.createElement('div');
                d.className = 'vrjs-ctl';
                const id = 'vrjs_' + k;
                d.innerHTML = '<label for="' + id + '">' + lab + '</label><input type="range" id="' + id + '" min="' + mn + '" max="' + mx + '" step="' + st + '"><output></output>';
                const r = d.querySelector('input'), o = d.querySelector('output');
                const show = v => { r.value = v; o.textContent = (k === 'rate' && v >= 60) ? 'max' : (k === 'levels' && v < 2) ? '∞' : (k === 'res' && v < 1) ? 'native' : v; };
                show(this.cfg[k]);
                this._ui[k] = show;
                r.oninput = () => this.patchConfig(k, r.value);
                host.appendChild(d);
            };
            const check = (host, k, lab) => {
                const d = document.createElement('div');
                d.className = 'vrjs-chk';
                const id = 'vrjs_' + k;
                d.innerHTML = '<input type="checkbox" id="' + id + '"><label for="' + id + '">' + lab + '</label>';
                const c = d.querySelector('input');
                c.checked = this.cfg[k];
                this._ui[k] = v => { c.checked = v; };
                c.onchange = () => this.patchConfig(k, c.checked);
                host.appendChild(d);
            };
            const section = (title, open) => {
                const d = document.createElement('details');
                if (open) d.open = true;
                d.innerHTML = '<summary>' + title + '</summary>';
                root.appendChild(d);
                return d;
            };
            const a = section('Haptic simulation', true);
            SIM.forEach(s => slider(a, s));
            const b = section('Headset layout and lens warp', false);
            check(b, 'test', 'Show test grid (all cells on)');
            check(b, 'smooth', 'Smooth edges between cells');
            LAY.forEach(s => slider(b, s));
            const c = section('Orientation', false);
            check(c, 'flipH', 'Flip left/right');
            check(c, 'flipV', 'Flip up/down');
            check(c, 'swap', 'Swap rows and columns');
            check(c, 'rotate', 'Rotate VR view 90° (for a phone kept in portrait; uses tilt sensor)');
            check(c, 'rotFlip', 'Reverse the rotation direction');
            const reset = document.createElement('button');
            reset.type = 'button';
            reset.className = 'vrjs-btn';
            reset.textContent = 'Reset all display settings';
            reset.onclick = () => { for (const k of Object.keys(DEFAULTS)) this.patchConfig(k, DEFAULTS[k]); };
            root.appendChild(reset);
            this._panels = root;
        }

        _openPopup() {
            if (this._pop) return;
            const pop = document.createElement('div');
            pop.className = 'vrjs-pop';
            const card = document.createElement('div');
            card.className = 'vrjs-card';
            const head = document.createElement('div');
            head.className = 'vrjs-head';
            head.innerHTML = '<span>Display settings</span>';
            const x = document.createElement('button');
            x.type = 'button';
            x.className = 'vrjs-btn';
            x.textContent = 'Close';
            x.onclick = () => this._closePopup();
            head.appendChild(x);
            card.appendChild(head);
            card.appendChild(this.configPanelsNode);
            pop.appendChild(card);
            pop.addEventListener('click', e => { if (e.target === pop) this._closePopup(); });
            document.body.appendChild(pop);
            this._pop = pop;
        }
        _closePopup() { if (this._pop) { this._pop.remove(); this._pop = null; } }

        /* ---------------- internals: data ---------------- */
        _push(grid) {
            const h = Array.isArray(grid) ? grid.length : 0;
            const w = h && Array.isArray(grid[0]) ? grid[0].length : 0;
            if (!w || !grid.every(r => Array.isArray(r) && r.length === w)) {
                console.error('VR.paint: expected a non-empty rectangular 2D array of numbers');
                return;
            }
            const v = new Float32Array(w * h);
            for (let y = 0; y < h; y++) for (let x = 0; x < w; x++) {
                const n = grid[y][x];
                v[y * w + x] = (typeof n === 'number' && isFinite(n)) ? n : NaN;
            }
            this._hist.push({ t: performance.now(), w, h, v });
            if (this._hist.length > 200) this._hist.shift();
        }

        _process(f, dt) {
            const c = this.cfg;
            const sw = c.swap, ow = sw ? f.h : f.w, oh = sw ? f.w : f.h;
            let t = new Float32Array(ow * oh);
            if (c.test) { t.fill(1); }
            else for (let y = 0; y < oh; y++) for (let x = 0; x < ow; x++) {
                const sy = c.flipV ? oh - 1 - y : y, sx = c.flipH ? ow - 1 - x : x;
                const v = sw ? f.v[sx * f.w + sy] : f.v[sy * f.w + sx];
                t[y * ow + x] = Number.isFinite(v) ? Math.min(1, Math.max(0, v)) : 0;
            }
            const m = Math.max(ow, oh), n = c.res;
            if (n > 0 && n < m) {
                const ny = Math.max(1, Math.round(n * oh / m)), nx = Math.max(1, Math.round(n * ow / m));
                const sum = new Float32Array(nx * ny), cnt = new Float32Array(nx * ny);
                const K = (y, x) => Math.min(ny - 1, (y * ny / oh) | 0) * nx + Math.min(nx - 1, (x * nx / ow) | 0);
                for (let y = 0; y < oh; y++) for (let x = 0; x < ow; x++) { const k = K(y, x); sum[k] += t[y * ow + x]; cnt[k]++; }
                for (let y = 0; y < oh; y++) for (let x = 0; x < ow; x++) { const k = K(y, x); t[y * ow + x] = sum[k] / cnt[k]; }
            }
            if (c.blur > 0) {
                const r = c.blur, o = new Float32Array(ow * oh);
                for (let y = 0; y < oh; y++) for (let x = 0; x < ow; x++) {
                    let s = 0, q = 0;
                    for (let j = -r; j <= r; j++) for (let i = -r; i <= r; i++) {
                        const yy = y + j, xx = x + i;
                        if (yy >= 0 && yy < oh && xx >= 0 && xx < ow) { s += t[yy * ow + xx]; q++; }
                    }
                    o[y * ow + x] = s / q;
                }
                t = o;
            }
            if (this._ad.length !== t.length) this._ad = new Float32Array(t.length);
            const b = new Uint8Array(t.length);
            for (let i = 0; i < t.length; i++) {
                let v = t[i];
                if (c.noise > 0) v += (Math.random() - 0.5) * 2 * c.noise;
                v = Math.min(1, Math.max(0, v));
                if (c.levels >= 2) v = Math.round(v * (c.levels - 1)) / (c.levels - 1);
                if (c.tau > 0) {
                    this._ad[i] = Math.min(0.95, Math.max(0, this._ad[i] + (v > 0.05 ? dt / c.tau : -dt / (c.tau * 0.5))));
                    v *= 1 - this._ad[i];
                } else this._ad[i] = 0;
                b[i] = Math.round(v * 255);
            }
            return { w: ow, h: oh, b };
        }

        /* ---------------- internals: portrait-mode rotation ---------------- */
        // Phone stays in portrait; the canvas is rotated 90 degrees in CSS so the view looks
        // landscape. Direction comes from the sign of gravity along the device x axis
        // (x > 0: rotate clockwise, x < 0: counter-clockwise). If it is backwards on your
        // phone, tick "Reverse the rotation direction" or swap `cw` below.
        _startMotion() {
            if (this._onMotion || !('DeviceMotionEvent' in window)) return;
            this._onMotion = e => {
                const a = e.accelerationIncludingGravity;
                if (!a || a.x == null || Math.abs(a.x) < 2) return; // ignore near-flat readings
                const d = a.x > 0 ? 1 : -1;
                if (d !== this._dir) { this._dir = d; this._resize(); }
            };
            window.addEventListener('devicemotion', this._onMotion);
        }
        _stopMotion() {
            if (this._onMotion) { window.removeEventListener('devicemotion', this._onMotion); this._onMotion = null; }
        }
        get rotatedDirection() { return (this._dir > 0) ? 1 : 0; }
        _applyRot() {
            const s = this._canvas.style, W = window.innerWidth, H = window.innerHeight;
            if (this._vr && this.cfg.rotate && H > W) { // only when the viewport really is portrait
                const cw = (this._dir > 0) !== this.cfg.rotFlip;
                s.left = '0'; s.top = '0';
                s.width = H + 'px'; s.height = W + 'px';
                s.transformOrigin = '0 0';
                s.transform = cw ? 'translateX(' + W + 'px) rotate(90deg)' : 'translateY(' + H + 'px) rotate(-90deg)';
            } else {
                s.left = s.top = s.width = s.height = s.transformOrigin = s.transform = '';
            }
        }

        /* ---------------- internals: WebGL ---------------- */
        _initGL() {
            const gl = this._gl = this._canvas.getContext('webgl');
            if (!gl) { console.error('VR: WebGL is not available'); return; }
            const sh = (t, s) => { const o = gl.createShader(t); gl.shaderSource(o, s); gl.compileShader(o); return o; };
            const pr = this._pr = gl.createProgram();
            gl.attachShader(pr, sh(gl.VERTEX_SHADER, VS));
            gl.attachShader(pr, sh(gl.FRAGMENT_SHADER, FS));
            gl.linkProgram(pr);
            gl.useProgram(pr);
            gl.bindBuffer(gl.ARRAY_BUFFER, gl.createBuffer());
            gl.bufferData(gl.ARRAY_BUFFER, new Float32Array([-1, -1, 1, -1, -1, 1, 1, 1]), gl.STATIC_DRAW);
            const loc = gl.getAttribLocation(pr, 'a');
            gl.enableVertexAttribArray(loc);
            gl.vertexAttribPointer(loc, 2, gl.FLOAT, false, 0, 0);
            gl.bindTexture(gl.TEXTURE_2D, gl.createTexture());
            gl.pixelStorei(gl.UNPACK_ALIGNMENT, 1);
            gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_S, gl.CLAMP_TO_EDGE);
            gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_T, gl.CLAMP_TO_EDGE);
            this._u = {};
            for (const n of ['res', 'dims', 'aspect', 'k1', 'k2', 'size', 'shift', 'voff', 'gap']) this._u[n] = gl.getUniformLocation(pr, n);
        }

        _resize() {
            this._applyRot();
            const cv = this._canvas;
            if (!this._gl || !cv.clientWidth || !cv.clientHeight) return;
            const dpr = Math.min(2, window.devicePixelRatio || 1);
            const w = Math.round(cv.clientWidth * dpr), h = Math.round(cv.clientHeight * dpr);
            if (cv.width !== w || cv.height !== h) { cv.width = w; cv.height = h; }
            this._gl.viewport(0, 0, cv.width, cv.height);
        }

        _draw() {
            const gl = this._gl, c = this.cfg, img = this._img;
            if (this._dirty) {
                const f = c.smooth ? gl.LINEAR : gl.NEAREST;
                gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MIN_FILTER, f);
                gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MAG_FILTER, f);
                gl.texImage2D(gl.TEXTURE_2D, 0, gl.LUMINANCE, img.w, img.h, 0, gl.LUMINANCE, gl.UNSIGNED_BYTE, img.b);
                this._dirty = false;
            }
            const u = this._u;
            gl.uniform2f(u.res, this._canvas.width, this._canvas.height);
            gl.uniform2f(u.dims, img.w, img.h);
            gl.uniform1f(u.aspect, img.w / img.h);
            for (const k of ['k1', 'k2', 'size', 'shift', 'voff', 'gap']) gl.uniform1f(u[k], c[k]);
            gl.drawArrays(gl.TRIANGLE_STRIP, 0, 4);
        }

        _loop(now) {
            requestAnimationFrame(t => this._loop(t));
            if (!this._gl || !this._canvas.isConnected) return;
            const c = this.cfg;
            const target = now - c.lag;
            let f = null;
            for (let i = this._hist.length - 1; i >= 0; i--) if (this._hist[i].t <= target) { f = this._hist[i]; break; }
            if (!f && c.test) f = { w: 8, h: 8, v: new Float32Array(64) };
            const gap = c.rate >= 60 ? 0 : 1000 / c.rate;
            if (f && now - this._lastUp >= gap) {
                this._img = this._process(f, Math.min(0.5, (now - this._lastUp) / 1000));
                this._lastUp = now;
                this._dirty = true;
            }
            this._draw();
        }
    }

    window.VRDisplay = VRDisplay;
    window.VR = new VRDisplay();
})();
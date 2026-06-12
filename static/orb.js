/**
 * VoiceOrb — a WebGL fragment-shader plasma orb for the voice assistant.
 *
 * Technique (inspired by the ElevenLabs / 21st.dev "Voice Powered Orb"):
 *   - 2D simplex noise + domain-warped fbm for the fluid plasma surface
 *   - fresnel-style rim light and a soft outer glow halo
 *   - real-time audio level (0..1) drives turbulence, brightness and the rim
 *   - per-state colour palettes (idle / thinking / speaking / listening),
 *     smoothly interpolated so transitions never snap
 *
 * Public API (window.VoiceOrb):
 *   init(canvas)        bind to a <canvas>; starts the render loop
 *   setState(name)      'idle' | 'thinking' | 'speaking' | 'listening'
 *   setLevel(v)         feed audio amplitude 0..1 (call each animation frame)
 */
(function () {
  'use strict';

  const VERT = `
    attribute vec2 p;
    void main () { gl_Position = vec4(p, 0.0, 1.0); }
  `;

  const FRAG = `
    precision highp float;
    uniform vec2  uRes;
    uniform float uTime;
    uniform float uLevel;
    uniform vec3  uColA;
    uniform vec3  uColB;
    uniform vec3  uColC;
    uniform vec3  uRim;

    // --- Simplex noise 2D (Ashima Arts / Stefan Gustavson, public domain) ---
    vec3 mod289(vec3 x){ return x - floor(x * (1.0/289.0)) * 289.0; }
    vec2 mod289(vec2 x){ return x - floor(x * (1.0/289.0)) * 289.0; }
    vec3 permute(vec3 x){ return mod289(((x*34.0)+1.0)*x); }
    float snoise(vec2 v){
      const vec4 C = vec4(0.211324865405187, 0.366025403784439,
                         -0.577350269189626, 0.024390243902439);
      vec2 i  = floor(v + dot(v, C.yy));
      vec2 x0 = v - i + dot(i, C.xx);
      vec2 i1 = (x0.x > x0.y) ? vec2(1.0, 0.0) : vec2(0.0, 1.0);
      vec4 x12 = x0.xyxy + C.xxzz;
      x12.xy -= i1;
      i = mod289(i);
      vec3 perm = permute( permute( i.y + vec3(0.0, i1.y, 1.0))
                                  + i.x + vec3(0.0, i1.x, 1.0));
      vec3 m = max(0.5 - vec3(dot(x0,x0), dot(x12.xy,x12.xy), dot(x12.zw,x12.zw)), 0.0);
      m = m*m; m = m*m;
      vec3 x = 2.0 * fract(perm * C.www) - 1.0;
      vec3 h = abs(x) - 0.5;
      vec3 ox = floor(x + 0.5);
      vec3 a0 = x - ox;
      m *= 1.79284291400159 - 0.85373472095314 * (a0*a0 + h*h);
      vec3 g;
      g.x  = a0.x  * x0.x  + h.x  * x0.y;
      g.yz = a0.yz * x12.xz + h.yz * x12.yw;
      return 130.0 * dot(m, g);
    }
    float fbm(vec2 p){
      float s = 0.0, a = 0.5;
      for (int i = 0; i < 5; i++){ s += a * snoise(p); p *= 2.0; a *= 0.5; }
      return s;
    }

    void main () {
      vec2 uv = (gl_FragCoord.xy - 0.5 * uRes) / min(uRes.x, uRes.y);
      float r = length(uv);
      float radius = 0.40;

      float lvl = clamp(uLevel, 0.0, 1.0);
      float t = uTime * (0.12 + lvl * 0.10);

      // Domain-warped fbm gives the swirling "fluid" plasma.
      vec2 q = uv * (2.0 + lvl * 0.6);
      float w = fbm(q + vec2(t, -t));
      float n = fbm(q + (1.3 + lvl) * vec2(w, w) + vec2(-t * 1.2, t * 0.8));
      float shade = clamp(n * 0.5 + 0.5, 0.0, 1.0);

      float sphere = smoothstep(radius, radius - 0.045, r);
      float rim    = smoothstep(radius - 0.06, radius - 0.005, r);

      vec3 col = mix(uColA, uColB, shade);
      col = mix(col, uColC, smoothstep(0.12, 0.52, r));

      // Soft key light from the upper-left for volume.
      float light = clamp(1.0 - length(uv - vec2(-0.13, 0.14)) * 1.7, 0.0, 1.0);
      col += light * 0.30;

      col *= (0.72 + lvl * 0.9 + shade * 0.25);
      col += rim * uRim * (0.5 + lvl * 1.2);

      // Outer glow halo beyond the sphere edge.
      float glow  = exp(-pow(max(r - radius, 0.0) * 6.5, 2.0));
      vec3  outer = uRim * glow * (0.30 + lvl * 0.75);

      vec3  finalCol = col * sphere + outer;
      float alpha    = clamp(max(sphere, glow * 0.9), 0.0, 1.0);
      gl_FragColor = vec4(finalCol, alpha);
    }
  `;

  // Colour palettes per state: [A, B, C, rim, idleBase, speed]
  const STATES = {
    idle:      { A: [0.00, 0.85, 1.00], B: [0.55, 0.30, 0.95], C: [0.02, 0.05, 0.13], rim: [0.10, 0.85, 1.00], base: 0.05, speed: 1.0 },
    thinking:  { A: [0.62, 0.32, 0.96], B: [0.20, 0.55, 1.00], C: [0.05, 0.04, 0.16], rim: [0.66, 0.36, 1.00], base: 0.22, speed: 1.9 },
    speaking:  { A: [0.00, 0.92, 1.00], B: [0.35, 0.70, 1.00], C: [0.03, 0.10, 0.26], rim: [0.20, 0.95, 1.00], base: 0.10, speed: 1.5 },
    listening: { A: [1.00, 0.27, 0.38], B: [1.00, 0.50, 0.62], C: [0.14, 0.02, 0.07], rim: [1.00, 0.32, 0.42], base: 0.16, speed: 1.5 },
  };

  function compile(gl, type, src) {
    const s = gl.createShader(type);
    gl.shaderSource(s, src);
    gl.compileShader(s);
    if (!gl.getShaderParameter(s, gl.COMPILE_STATUS)) {
      console.error('[Orb] shader error:', gl.getShaderInfoLog(s));
      return null;
    }
    return s;
  }

  const lerp3 = (a, b, t) => [a[0]+(b[0]-a[0])*t, a[1]+(b[1]-a[1])*t, a[2]+(b[2]-a[2])*t];

  const VoiceOrb = {
    ok: false,
    init(canvas) {
      if (!canvas) return false;
      this.canvas = canvas;
      const gl = canvas.getContext('webgl', { alpha: true, premultipliedAlpha: false, antialias: true })
              || canvas.getContext('experimental-webgl', { alpha: true });
      if (!gl) { console.warn('[Orb] WebGL unavailable'); return false; }
      this.gl = gl;

      const vs = compile(gl, gl.VERTEX_SHADER, VERT);
      const fs = compile(gl, gl.FRAGMENT_SHADER, FRAG);
      if (!vs || !fs) return false;
      const prog = gl.createProgram();
      gl.attachShader(prog, vs);
      gl.attachShader(prog, fs);
      gl.linkProgram(prog);
      if (!gl.getProgramParameter(prog, gl.LINK_STATUS)) {
        console.error('[Orb] link error:', gl.getProgramInfoLog(prog));
        return false;
      }
      gl.useProgram(prog);
      this.prog = prog;

      const buf = gl.createBuffer();
      gl.bindBuffer(gl.ARRAY_BUFFER, buf);
      gl.bufferData(gl.ARRAY_BUFFER, new Float32Array([-1,-1, 3,-1, -1,3]), gl.STATIC_DRAW);
      const loc = gl.getAttribLocation(prog, 'p');
      gl.enableVertexAttribArray(loc);
      gl.vertexAttribPointer(loc, 2, gl.FLOAT, false, 0, 0);

      this.u = {
        res:   gl.getUniformLocation(prog, 'uRes'),
        time:  gl.getUniformLocation(prog, 'uTime'),
        level: gl.getUniformLocation(prog, 'uLevel'),
        colA:  gl.getUniformLocation(prog, 'uColA'),
        colB:  gl.getUniformLocation(prog, 'uColB'),
        colC:  gl.getUniformLocation(prog, 'uColC'),
        rim:   gl.getUniformLocation(prog, 'uRim'),
      };

      gl.enable(gl.BLEND);
      gl.blendFunc(gl.SRC_ALPHA, gl.ONE_MINUS_SRC_ALPHA);

      // Current (smoothed) palette + targets.
      const s = STATES.idle;
      this.cur = { A: s.A.slice(), B: s.B.slice(), C: s.C.slice(), rim: s.rim.slice() };
      this.target = s;
      this.level = 0;
      this.targetLevel = 0;
      this.speed = 1;
      this.time = 0;
      this.ok = true;

      this.resize();
      window.addEventListener('resize', () => this.resize());
      this._last = performance.now();
      requestAnimationFrame(this._loop.bind(this));
      return true;
    },

    resize() {
      if (!this.ok) return;
      const dpr = Math.min(window.devicePixelRatio || 1, 2);
      const css = this.canvas.clientWidth || 210;
      const px = Math.round(css * dpr);
      if (this.canvas.width !== px) {
        this.canvas.width = px;
        this.canvas.height = px;
      }
      this.gl.viewport(0, 0, this.canvas.width, this.canvas.height);
    },

    setState(name) {
      const s = STATES[name];
      if (!s || !this.ok) return;
      this.target = s;
      this.targetLevel = 0;   // audio (if any) will drive it back up
    },

    setLevel(v) {
      this.targetLevel = Math.max(0, Math.min(1, v || 0));
    },

    _loop(now) {
      if (!this.ok) return;
      const dt = Math.min((now - this._last) / 1000, 0.05);
      this._last = now;

      const k = 1 - Math.pow(0.001, dt);   // ~frame-rate independent smoothing
      this.cur.A   = lerp3(this.cur.A,   this.target.A,   k);
      this.cur.B   = lerp3(this.cur.B,   this.target.B,   k);
      this.cur.C   = lerp3(this.cur.C,   this.target.C,   k);
      this.cur.rim = lerp3(this.cur.rim, this.target.rim, k);
      this.speed  += (this.target.speed - this.speed) * k;

      // Effective level never drops below the state's idle shimmer.
      const want = Math.max(this.targetLevel, this.target.base);
      const aim  = want > this.level ? 0.35 : 0.06;   // fast attack, slow release
      this.level += (want - this.level) * aim;

      this.time += dt * this.speed;

      const gl = this.gl, u = this.u;
      gl.uniform2f(u.res, this.canvas.width, this.canvas.height);
      gl.uniform1f(u.time, this.time);
      gl.uniform1f(u.level, this.level);
      gl.uniform3fv(u.colA, this.cur.A);
      gl.uniform3fv(u.colB, this.cur.B);
      gl.uniform3fv(u.colC, this.cur.C);
      gl.uniform3fv(u.rim, this.cur.rim);

      gl.clearColor(0, 0, 0, 0);
      gl.clear(gl.COLOR_BUFFER_BIT);
      gl.drawArrays(gl.TRIANGLES, 0, 3);

      requestAnimationFrame(this._loop.bind(this));
    },
  };

  window.VoiceOrb = VoiceOrb;
})();

(function () {
  "use strict";
  // Two-hill model: A(d, o) = 1/2 [Phi(d/2 + o) + Phi(d/2 - o)].
  // The 2D picture uses two round Gaussians (unit variance) at (-D/2, 0) and (+D/2, 0).
  // A classifier direction tilted by theta sees separation d = D cos(theta).
  var D = 4.0;
  var DEFAULT = { theta: Math.acos(0.8) * 180 / Math.PI, o: 0.8 };  // d = 3.2, o = 0.8

  function erf(x) {  // Abramowitz-Stegun 7.1.26, |error| < 1.5e-7
    var s = x < 0 ? -1 : 1; x = Math.abs(x);
    var t = 1 / (1 + 0.3275911 * x);
    var y = 1 - (((((1.061405429 * t - 1.453152027) * t) + 1.421413741) * t - 0.284496736) * t + 0.254829592) * t * Math.exp(-x * x);
    return s * y;
  }
  function Phi(z) { return 0.5 * (1 + erf(z / Math.SQRT2)); }
  function phi(z) { return Math.exp(-z * z / 2) / Math.sqrt(2 * Math.PI); }
  function A(d, o) { return 0.5 * (Phi(d / 2 + o) + Phi(d / 2 - o)); }

  function state(theta, o) {
    var e = Math.cos(theta * Math.PI / 180), d = D * e;
    var a = A(d, o), a0 = A(d, 0), aD = A(D, 0);
    return { theta: theta, o: o, e: e, d: d, a: a, a0: a0, aD: aD,
      pos: 100 * (a0 - a), dir: 100 * (aD - a0), rest: 100 * (1 - aD) };
  }

  // Deterministic sample of points (mulberry32 + Box-Muller).
  function rng(seed) {
    return function () {
      seed = (seed + 0x6D2B79F5) | 0;
      var t = Math.imul(seed ^ (seed >>> 15), 1 | seed);
      t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
      return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
    };
  }
  var R = rng(11), PTS = [];
  function gauss() { var u = 0; while (u === 0) u = R(); var v = R(); return Math.sqrt(-2 * Math.log(u)) * Math.cos(2 * Math.PI * v); }
  for (var c = 0; c < 2; c++) for (var i = 0; i < 60; i++) PTS.push({ c: c, x: (c ? 1 : -1) * D / 2 + gauss(), y: gauss() });

  function f1(v) { return (Math.round(v * 10) / 10).toFixed(1); }
  function f2(v) { return v.toFixed(2); }

  // ---- left panel: the plane -------------------------------------------
  var PX = { W: 320, H: 262, cx: 160, cy: 130, s: 32, xmin: -4.6, xmax: 4.6, ymin: -3.5, ymax: 3.5 };
  function X(x) { return PX.cx + PX.s * x; }
  function Y(y) { return PX.cy - PX.s * y; }
  function clipLine(theta, o) {
    var th = theta * Math.PI / 180, vx = Math.cos(th), vy = Math.sin(th);
    var p0 = [o * vx, o * vy], u = [-vy, vx];
    var t0 = -50, t1 = 50;
    var lims = [[u[0], p0[0], PX.xmin, PX.xmax], [u[1], p0[1], PX.ymin, PX.ymax]];
    for (var k = 0; k < 2; k++) {
      var du = lims[k][0], p = lims[k][1], lo = lims[k][2], hi = lims[k][3];
      if (Math.abs(du) < 1e-12) { if (p < lo || p > hi) return null; continue; }
      var a = (lo - p) / du, b = (hi - p) / du;
      if (a > b) { var tmp = a; a = b; b = tmp; }
      t0 = Math.max(t0, a); t1 = Math.min(t1, b);
    }
    if (t0 >= t1) return null;
    return [[p0[0] + t0 * u[0], p0[1] + t0 * u[1]], [p0[0] + t1 * u[0], p0[1] + t1 * u[1]]];
  }
  function svgPlane(s) {
    var th = s.theta * Math.PI / 180, vx = Math.cos(th), vy = Math.sin(th);
    var out = [];
    out.push('<svg viewBox="0 0 ' + PX.W + ' ' + PX.H + '" role="img" aria-label="2 つの感情の発話の点と，f の境目（実線）と最良の境目（破線）">');
    out.push('<rect x="' + X(PX.xmin) + '" y="' + Y(PX.ymax) + '" width="' + (PX.s * (PX.xmax - PX.xmin)) + '" height="' + (PX.s * (PX.ymax - PX.ymin)) + '" class="frame"/>');
    PTS.forEach(function (p) {
      if (p.x < PX.xmin || p.x > PX.xmax || p.y < PX.ymin || p.y > PX.ymax) return;
      var z = vx * p.x + vy * p.y;
      var wrong = p.c === 0 ? z > s.o : z < s.o;
      out.push('<circle cx="' + X(p.x).toFixed(1) + '" cy="' + Y(p.y).toFixed(1) + '" r="3.3" class="pt ' + (p.c ? 'b' : 'a') + (wrong ? ' wrong' : '') + '"/>');
    });
    out.push('<line x1="' + X(0) + '" y1="' + Y(PX.ymax) + '" x2="' + X(0) + '" y2="' + Y(PX.ymin) + '" class="best"/>');
    out.push('<text x="' + X(0) + '" y="' + (Y(PX.ymin) + 15) + '" text-anchor="middle" class="lab">最良の境目</text>');
    var seg = clipLine(s.theta, s.o);
    if (seg) {
      var a = seg[0], b = seg[1];
      out.push('<line x1="' + X(a[0]).toFixed(1) + '" y1="' + Y(a[1]).toFixed(1) + '" x2="' + X(b[0]).toFixed(1) + '" y2="' + Y(b[1]).toFixed(1) + '" class="cut"/>');
      // label at the upper end, kept inside the frame
      var tx = X(b[0]), ty = Y(b[1]) + 13, anchor = 'start', lx = tx + 6;
      if (lx + 56 > PX.W - 4) { anchor = 'end'; lx = tx - 6; }
      out.push('<text x="' + lx.toFixed(1) + '" y="' + ty.toFixed(1) + '" text-anchor="' + anchor + '" class="lab strong">f の境目</text>');
    }
    out.push('</svg>');
    return out.join('');
  }

  // ---- right panel: the number line ------------------------------------
  var NL = { W: 360, H: 226, x0: 20, s: 32, zmin: -5, zmax: 5, base: 184, h: 320 };
  function Z(z) { return NL.x0 + (z - NL.zmin) * NL.s; }
  function curve(mu, from, to) {
    var pts = [], step = 0.05;
    for (var z = from; z <= to + 1e-9; z += step) pts.push(Z(z).toFixed(1) + ',' + (NL.base - NL.h * phi(z - mu)).toFixed(1));
    return pts;
  }
  function area(mu, from, to) {
    if (to <= from) return '';
    var pts = curve(mu, from, to);
    return 'M' + Z(from).toFixed(1) + ',' + NL.base + ' L' + pts.join(' L') + ' L' + Z(to).toFixed(1) + ',' + NL.base + ' Z';
  }
  function svgLine(s) {
    var o = Math.max(NL.zmin, Math.min(NL.zmax, s.o));
    var ma = -s.d / 2, mb = s.d / 2, out = [];
    out.push('<svg viewBox="0 0 ' + NL.W + ' ' + NL.H + '" role="img" aria-label="f の点数の差の数直線に並んだ 2 つの山と境目">');
    // ghosts: the best direction
    out.push('<polyline points="' + curve(-D / 2, NL.zmin, NL.zmax).join(' ') + '" class="ghost"/>');
    out.push('<polyline points="' + curve(D / 2, NL.zmin, NL.zmax).join(' ') + '" class="ghost"/>');
    // hills
    out.push('<path d="' + area(ma, NL.zmin, NL.zmax) + '" class="hill a"/>');
    out.push('<path d="' + area(mb, NL.zmin, NL.zmax) + '" class="hill b"/>');
    // errors: a (left emotion) right of the cut, b left of the cut
    out.push('<path d="' + area(ma, o, NL.zmax) + '" class="err a"/>');
    out.push('<path d="' + area(mb, NL.zmin, o) + '" class="err b"/>');
    out.push('<polyline points="' + curve(ma, NL.zmin, NL.zmax).join(' ') + '" class="edge a"/>');
    out.push('<polyline points="' + curve(mb, NL.zmin, NL.zmax).join(' ') + '" class="edge b"/>');
    // axis
    out.push('<line x1="' + Z(NL.zmin) + '" y1="' + NL.base + '" x2="' + Z(NL.zmax) + '" y2="' + NL.base + '" class="axis"/>');
    [-4, -2, 0, 2, 4].forEach(function (t) {
      out.push('<line x1="' + Z(t) + '" y1="' + NL.base + '" x2="' + Z(t) + '" y2="' + (NL.base + 4) + '" class="axis"/>');
      out.push('<text x="' + Z(t) + '" y="' + (NL.base + 17) + '" text-anchor="middle" class="tick">' + (t < 0 ? '−' + (-t) : t) + '</text>');
    });
    out.push('<text x="' + Z(0) + '" y="' + (NL.base + 36) + '" text-anchor="middle" class="tick">f の点数の差（山の幅を 1 とする目盛り）</text>');
    // middle and cut
    var top = NL.base - NL.h * phi(0) - 6;
    out.push('<line x1="' + Z(0) + '" y1="22" x2="' + Z(0) + '" y2="' + NL.base + '" class="best"/>');
    out.push('<text x="' + Z(0) + '" y="16" text-anchor="middle" class="lab">真ん中</text>');
    out.push('<line x1="' + Z(o) + '" y1="40" x2="' + Z(o) + '" y2="' + NL.base + '" class="cut"/>');
    out.push('<text x="' + Z(o) + '" y="34" text-anchor="middle" class="lab strong">f の境目</text>');
    // hill labels
    var xa = Z(ma), xb = Z(mb), ancA = 'middle', ancB = 'middle';
    if (xb - xa < 64) { xa -= 4; xb += 4; ancA = 'end'; ancB = 'start'; }
    out.push('<text x="' + xa.toFixed(1) + '" y="' + top.toFixed(1) + '" text-anchor="' + ancA + '" class="lab">悲しみ</text>');
    out.push('<text x="' + xb.toFixed(1) + '" y="' + top.toFixed(1) + '" text-anchor="' + ancB + '" class="lab">平静</text>');
    out.push('</svg>');
    return out.join('');
  }

  // ---- bottom: the lost points -----------------------------------------
  var BR = { W: 640, H: 60, x0: 12, s: 12.3, y: 6, h: 24 };
  function svgBar(s) {
    var out = [], x = BR.x0, gap = 2;
    out.push('<svg viewBox="0 0 ' + BR.W + ' ' + BR.H + '" role="img" aria-label="失った正解率の内訳：位置の損 ' + f1(s.pos) + ' 点，方向の損 ' + f1(s.dir) + ' 点，最良の向きでも残る誤り ' + f1(s.rest) + ' 点">');
    [0, 10, 20, 30, 40, 50].forEach(function (t) {
      out.push('<line x1="' + (BR.x0 + t * BR.s) + '" y1="' + (BR.y - 2) + '" x2="' + (BR.x0 + t * BR.s) + '" y2="' + (BR.y + BR.h + 4) + '" class="grid"/>');
      out.push('<text x="' + (BR.x0 + t * BR.s) + '" y="' + (BR.y + BR.h + 18) + '" text-anchor="middle" class="tick">' + t + '</text>');
    });
    [['pos', s.pos], ['dir', s.dir], ['rest', s.rest]].forEach(function (p) {
      var w = Math.max(0, p[1]) * BR.s;
      if (w <= 0) return;
      var drawn = Math.max(0.5, w - gap);
      out.push('<rect x="' + x.toFixed(1) + '" y="' + BR.y + '" width="' + drawn.toFixed(1) + '" height="' + BR.h + '" rx="3" class="seg ' + p[0] + '"><title>' + { pos: '位置の損', dir: '方向の損', rest: '最良の向きでも残る誤り' }[p[0]] + ' ' + f1(p[1]) + ' 点</title></rect>');
      x += w;
    });
    out.push('</svg>');
    return out.join('');
  }

  var api = { D: D, DEFAULT: DEFAULT, A: A, state: state, svgPlane: svgPlane, svgLine: svgLine, svgBar: svgBar, f1: f1, f2: f2 };
  if (typeof module !== 'undefined' && module.exports) { module.exports = api; return; }

  // ---- page wiring ----------------------------------------------------
  var root = document.getElementById('hills');
  if (!root) return;
  var sT = root.querySelector('#hills-theta'), sO = root.querySelector('#hills-o');
  function q(id) { return root.querySelector('[data-v="' + id + '"]'); }
  function render() {
    var s = state(parseFloat(sT.value), parseFloat(sO.value));
    root.querySelector('.hills-plane').innerHTML = svgPlane(s);
    root.querySelector('.hills-line').innerHTML = svgLine(s);
    root.querySelector('.hills-bar').innerHTML = svgBar(s);
    q('theta').textContent = f1(s.theta) + '°';
    q('o').textContent = f2(s.o);
    q('d').textContent = f2(s.d);
    q('e').textContent = f2(s.e);
    q('a').textContent = f1(100 * s.a);
    q('a0').textContent = f1(100 * s.a0);
    q('aD').textContent = f1(100 * s.aD);
    q('pos').textContent = f1(s.pos);
    q('dir').textContent = f1(s.dir);
    q('rest').textContent = f1(s.rest);
    q('lost').textContent = f1(100 - 100 * s.a);
  }
  sT.addEventListener('input', render);
  sO.addEventListener('input', render);
  root.querySelector('#hills-reset').addEventListener('click', function () {
    sT.value = DEFAULT.theta.toFixed(1); sO.value = DEFAULT.o; render();
  });
  sT.value = DEFAULT.theta.toFixed(1); sO.value = DEFAULT.o;
  render();
})();

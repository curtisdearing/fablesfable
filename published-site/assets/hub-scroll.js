/* FablesFable hub scroll bridge.
 * When this page is embedded by the Dearing Football hub, report the page's
 * scroll position (numbers only) so the hub can tuck its header away on small
 * screens while the reader scrolls. Does nothing when opened directly or when
 * embedded by any other site. The hub says hello from its own origin; only an
 * allowlisted parent window receives messages, never '*'. */
(function () {
  'use strict';
  if (window.parent === window) return;
  var ALLOWED = /^(?:https:\/\/(?:www\.)?dearing-wedding\.com|http:\/\/(?:localhost|127\.0\.0\.1)(?::\d{1,5})?)$/;
  var target = null;
  var queued = false;

  function send() {
    queued = false;
    if (!target) return;
    var doc = document.scrollingElement || document.documentElement;
    var y = Math.max(0, Math.round(window.scrollY || doc.scrollTop || 0));
    var max = Math.max(0, Math.round(doc.scrollHeight - window.innerHeight));
    window.parent.postMessage({ type: 'dearing-hub:scroll', v: 1, y: y, max: max }, target);
  }
  function schedule() {
    if (!queued) { queued = true; window.requestAnimationFrame(send); }
  }

  window.addEventListener('message', function (e) {
    var d = e.data;
    if (e.source !== window.parent || !ALLOWED.test(e.origin)) return;
    if (!d || d.type !== 'dearing-hub:hello') return;
    target = e.origin;
    schedule();
  });
  window.addEventListener('scroll', schedule, { passive: true });
})();

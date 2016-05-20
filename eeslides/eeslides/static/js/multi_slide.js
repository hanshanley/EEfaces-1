var vids = [];
var slider;
var count = 0;

var w = window,
    d = document,
    e = d.documentElement,
    g = d.getElementsByTagName('body')[0],
    x = w.innerWidth || e.clientWidth || g.clientWidth,
    y = w.innerHeight || e.clientHeight || g.clientHeight;

function prepVideo(id) {
    var place = "holder_" + id;
    swfobject.embedSWF("http://www.youtube.com/v/" + id + "?enablejsapi=1&playerapiid=player_" + id + "&version=3&cc_load_policy=1&controls=1",
        place, x, y, "8", null, null, { allowScriptAccess: "always", allowfullscreen: "true" }, {id: id});
}
function onYouTubePlayerReady(playerId) {
    eleId = playerId.substr(7);
    window[playerId] = document.getElementById(eleId);
    window[playerId].addEventListener("onStateChange", "onPlayerStateChange");

    slider.redrawSlider();
}
function onPlayerStateChange(event) {
    if (event == 0) {
        slider.startAuto();
        slider.goToNextSlide();
    }
}
$(document).ready(function () {
    slider = $('.slide-container').bxSlider({
        auto: true,
        pause: 10000,
        controls: false,
        pager: false,
        onSlideAfter: function (e) {
            eid = $(e).attr('id').substr(6)
            if (vids.indexOf(eid) >= 0) {
                window['player_' + eid].playVideo()
                slider.stopAuto()
            }
            else if (slider.getCurrentSlide() == 0) {
                if (count > 4) { 
                    //window.location = "http://ee.princeton.edu/slider/"
                    window.location = "http://localhost:8000/"
                }
                count++;
            }
        }
    });
});

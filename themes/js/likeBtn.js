var check_index = 0
var btn_state = true
var gradioDislikeBtn = document.querySelectorAll('.svelte-3snf3m')

for (var i = 0; i < gradioDislikeBtn.length; i++) {
    gradioDislikeBtn[i].onclick = (function (index) {
        return function () {
            check_index = index
            if (btn_state) {
                this.style.color = '#F5BA2F'
            } else {
                this.style.color = '#D3D3D3'
            }
            btn_state = !btn_state

            if ((check_index % 2) == 0) {
                gradioDislikeBtn[index].style.color = "#F5BA2F"
                gradioDislikeBtn[index + 1].style.color = "#D3D3D3"
            } else {
                gradioDislikeBtn[index].style.color = "#F5BA2F"
                gradioDislikeBtn[index - 1].style.color = "#D3D3D3"
            }
        }
    })(i);
}
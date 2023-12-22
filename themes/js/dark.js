if (document.querySelectorAll('.dark').length) {
    document.querySelectorAll('.dark').forEach(el => el.classList.remove('dark'));
} else {
    document.querySelector('body').classList.add('dark');
}
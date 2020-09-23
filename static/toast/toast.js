const Toast = {
  init () {
    this.hideTimeout = null;

    this.el = document.createElement ('div');
    this.el.className = 'toast';
    document.getElementById('item-11').appendChild (this.el);
  },
  show (message) {

    var tweetinput = document.getElementById("UserInput").value;
    clearTimeout (this.hideTimeout);
    this.el.textContent = tweetinput;
    this.el.className = 'toast toast--visible';
  },

  showperm (message) {

    var tweetinput = message;
    clearTimeout (this.hideTimeout);
    this.el.textContent = tweetinput;
    this.el.className = 'toast toast--visible';
  }

};

document.addEventListener('DOMContentLoaded', () => Toast.init());

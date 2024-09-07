(function ($) {
    "use strict";

    //Preloader
    jQuery(window).on('load',function(){
        $(".preloader").delay(1600).fadeOut("slow");
    });
  
    // wow js
    new WOW({
        boxClass: 'wow', // default
        animateClass: 'animated', // default
        offset: 0, // default
        mobile: true, // default
        live: true // default
    }).init();

    // Custom datepickert
    $(".check-in").datepicker({
        dateFormat: "d MM yy",
        duration: "medium"
    });
    
    $(".check-out").datepicker({
        dateFormat: "d MM yy",
        duration: "medium"
    });

    // magnific-popup js
    $('.main-gallary').magnificPopup({
        type:'image',
        gallery:{
          enabled:true
        }
    });

    $('.video-icon').magnificPopup({
        type:'iframe',
        iframe: {
          markup: '<div class="mfp-iframe-scaler">'+
                    '<div class="mfp-close"></div>'+
                    '<iframe class="mfp-iframe" frameborder="0" allowfullscreen></iframe>'+
                  '</div>',
          patterns: {
            youtube: {
              index: 'youtube.com/',
              id: 'v=',
              src: '//www.youtube.com/embed/%id%?autoplay=1'
            },
            vimeo: {
              index: 'vimeo.com/',
              id: '/',
              src: '//player.vimeo.com/video/%id%?autoplay=1'
            },
            gmaps: {
              index: '//maps.google.',
              src: '%id%&output=embed'
            }
          },
          srcAction: 'iframe_src',
        }
    });

    //  custom select input
    var x, i, j, l, ll, selElmnt, a, b, c;
        x = document.getElementsByClassName("custom-select");
        l = x.length;
    for (i = 0; i < l; i++) {
        selElmnt = x[i].getElementsByTagName("select")[0];
        ll = selElmnt.length;
        a = document.createElement("DIV");
        a.setAttribute("class", "select-selected");
        a.innerHTML = selElmnt.options[selElmnt.selectedIndex].innerHTML;
        x[i].appendChild(a);
        b = document.createElement("DIV");
        b.setAttribute("class", "select-items select-hide");
        for (j = 1; j < ll; j++) {
        c = document.createElement("DIV");
        c.innerHTML = selElmnt.options[j].innerHTML;
        c.addEventListener("click", function(e) {
            var y, i, k, s, h, sl, yl;
            s = this.parentNode.parentNode.getElementsByTagName("select")[0];
            sl = s.length;
            h = this.parentNode.previousSibling;
            for (i = 0; i < sl; i++) {
              if (s.options[i].innerHTML == this.innerHTML) {
                s.selectedIndex = i;
                h.innerHTML = this.innerHTML;
                y = this.parentNode.getElementsByClassName("same-as-selected");
                yl = y.length;
                for (k = 0; k < yl; k++) {
                  y[k].removeAttribute("class");
                }
                this.setAttribute("class", "same-as-selected");
                break;
              }
            }
            h.click();
        });
        b.appendChild(c);
        }
        x[i].appendChild(b);
        a.addEventListener("click", function(e) {
          /*when the select box is clicked, close any other select boxes,
          and open/close the current select box:*/
          e.stopPropagation();
          closeAllSelect(this);
          this.nextSibling.classList.toggle("select-hide");
          this.classList.toggle("select-arrow-active");
        });
    }
    function closeAllSelect(elmnt) {
        var x, y, i, xl, yl, arrNo = [];
        x = document.getElementsByClassName("select-items");
        y = document.getElementsByClassName("select-selected");
        xl = x.length;
        yl = y.length;
        for (i = 0; i < yl; i++) {
            if (elmnt == y[i]) {
              arrNo.push(i)
            } else {
              y[i].classList.remove("select-arrow-active");
            }
        }
        for (i = 0; i < xl; i++) {
            if (arrNo.indexOf(i)) {
              x[i].classList.add("select-hide");
            }
        }
    }
    document.addEventListener("click", closeAllSelect);

    // mobile menu
    $('.hamburger').on('click',function (event) {
        $(this).toggleClass('h-active');
        $('.main-nav').toggleClass('slidenav');
    });

    $('.header-home .main-nav ul li  a').on('click',function (event) {
        $('.hamburger').removeClass('h-active');
        $('.main-nav').removeClass('slidenav');
    });

    $(".main-nav .fl").on('click', function(event) {
        var $fl = $(this);
        $(this).parent().siblings().find('.sub-menu').slideUp();
        $(this).parent().siblings().find('.fl').addClass('flaticon-plus').text("+");
    if($fl.hasClass('flaticon-plus')){
        $fl.removeClass('flaticon-plus').addClass('flaticon-minus').text("-");
    }else{
        $fl.removeClass('flaticon-minus').addClass('flaticon-plus').text("+");
    }
        $fl.next(".sub-menu").slideToggle();
    });


  //account dropdown
    var accountCard = document.querySelectorAll('.account-dropdown')
    var userIcon = document.querySelectorAll('.user-dropdown-icon i')

    userIcon.forEach((el)=>{
        el.addEventListener('click', ()=>{
          accountCard.forEach((element)=>{
            element.classList.toggle("activeCard")
          })
        })
    });
  
    // Search Bar js
    var searchOpen = document.querySelectorAll('.searchbar-open i')
    var searchCard = document.querySelectorAll('.main-searchbar')
    var searchClose = document.querySelectorAll('.searchbar-close i')

    searchOpen.forEach((el)=>{
        el.addEventListener('click',()=>{
          searchCard.forEach((el)=>{
            el.classList.add('activeSearch')
          })
        })
    })
    searchClose.forEach((el)=>{
        el.addEventListener('click',()=>{
          searchCard.forEach((el)=>{
            el.classList.remove('activeSearch')
          })
        })
    })
    window.onclick = function(event){
        searchCard.forEach((el)=>{
          if(event.target == el){
            el.classList.remove('activeSearch')
          }
        })
        if(!event.target.matches('.user-dropdown-icon i')){
          accountCard.forEach((element)=>{
            if(element.classList.contains('activeCard')){
              element.classList.remove('activeCard')
            }
          })
        }
    }
  

    // sticky navabr js
    $(window).on('scroll',function () {
        var scroll = $(window).scrollTop();
        if (scroll >= 10) {
            $(".header-area").addClass("sticky");
        } else {
            $(".header-area").removeClass("sticky");
        }
    });


    // banner slider
    $('.banner-slider').owlCarousel({
        items: 1,
        loop: true,
        margin:0,
        smartSpeed: 700,
        dots: false,
        nav: true,
        autoplay: 4000,
        autoplayTimeout:4000,
        autoplayHoverPause:true,
        animateIn: 'fadeIn',
        animateOut: 'fadeOut', 
        navText : ["<i class='bx bx-chevron-left' ></i>","<i class='bx bx-chevron-right'></i>"],
        responsive:{
            0:{
                items:1,
                nav:false,
                dots : false
            },
            600:{
                items:1,
                nav:false,
                dost : false,
            },
            1000:{
                items:1,
                nav:true,
                loop:true
            },
        }
    
    
    });
  
     // destinations-1 slider
     $('.destinations-1').owlCarousel({
         stagePadding: 1,
          items: 3,
          loop: true,
          margin:20,
          smartSpeed: 1500,
          autoplay: false,
          dots: false,
          nav: true,
          navText : ["<i class='bx bx-chevron-left' ></i>","<i class='bx bx-chevron-right'></i>"],
          responsive:{
              0:{
                  items:1,
                  nav:false,
                  dots : false
              },
              600:{
                  items:2,
                  nav:false,
                  dost : false,
              },
              1000:{
                  items:3,
                  nav:true,
                  loop:true
              }
          }
    });

    $('.destinations-2').owlCarousel({
        stagePadding: 1,
          items: 3,
          loop: true,
          margin:20,
          smartSpeed: 1500,
          autoplay: false,
          dots: false,
          nav: true,
          navText : ["<i class='bx bx-chevron-left' ></i>","<i class='bx bx-chevron-right'></i>"],
          responsive:{
              0:{
                  items:1,
                  nav:false,
                  dots : false
              },
              600:{
                  items:2,
                  nav:false,
                  dost : false,
              },
              1000:{
                  items:3,
                  nav:true,
                  loop:true
              }
          }
    });
    
    // review slider 
    $('.review-slider').owlCarousel({
            stagePadding: 10,
          items: 3,
          loop: true,
          margin:15,
          smartSpeed: 1500,
          autoplay: false,
          dots: true,
          nav: true,
          navText : ["<i class='bx bx-chevron-left' ></i>","<i class='bx bx-chevron-right'></i>"],
          responsive:{
              0:{
                  items:1,
                  nav:false,
                  dots : false
              },
              800:{
                  items:2,
                  nav:false,
                  dots : false,
              },
              1000:{
                  items:3,
                  dots: true,
                  nav:false,
                  loop:true
              }
          }
    });
    
    // feature slider
    $('.feature-slider').owlCarousel({
          items: 2,
          loop: true,
          margin:30,
          smartSpeed: 1500,
          autoplay: false,
          dots: false,
          nav: true,
          navText : ["<i class='bx bx-chevron-left' ></i>","<i class='bx bx-chevron-right'></i>"],
          responsive:{
              0:{
                  items:1,
                  nav:false,
                  dots : false
              },
              600:{
                  items:1,
                  nav:false,
                  dots : false,
              },
              1000:{
                  items:2,
                  dots: false,
                  nav: false,
                  loop:true
              }
          }
    });
    
    // offer slider
    $('.offer-slider').owlCarousel({
        stagePadding: 1,
        items: 3,
        loop: true,
        margin:25,
        smartSpeed: 1500,
        autoplay: false,
        dots: false,
        nav: true,
        navText : ["<i class='bx bx-chevron-left' ></i>","<i class='bx bx-chevron-right'></i>"],
        responsive:{
            0:{
                items:1,
                nav:false,
                dots : false
            },
            600:{
                items:2,
                nav:false,
                dots : false,
            },
            1000:{
                items:3,
                dots: false,
                nav:false,
                loop:true
            }
        }
    });

       // feature slider 2
    $('.feature-slider-2').owlCarousel({
        items: 3,
        loop: true,
        margin:25,
        smartSpeed: 1500,
        autoplay: false,
        dots: false,
        nav: true,
        animateOut: 'slideOutUp',
        animateIn: 'slideInUp',
        navText : ["<i class='bx bx-chevron-left' ></i>","<i class='bx bx-chevron-right'></i>"],
        responsive:{
            0:{
                items:1,
                nav:false,
                dots : false
            },
            800:{
                items:2,
                nav:false,
                dots : false,
            },
            1000:{
                items:3,
                dots: false,
                nav: true,
                loop:true
            }
        }
    });

        // feature slider 2
    $('.guide-slider').owlCarousel({
        items: 3,
        loop: true,
        margin:25,
        smartSpeed: 1500,
        autoplay: false,
        dots: false,
        nav: true,
        navText : ["<i class='bx bx-chevron-left' ></i>","<i class='bx bx-chevron-right'></i>"],
        responsive:{
            0:{
                items:1,
                nav:false,
                dots : false
            },
            600:{
                items:2,
                nav:false,
                dots : false,
            },
            1000:{
                items:3,
                dots: false,
                nav: true,
                loop:true
            }
        }
    });

    //Type js
    var element = $(".element");
    $(function () {
        element.typed({
            strings: ["Hampshire", "Indonesia", "Madagascar "],
            typeSpeed: 190,
            loop: true,
        });
    });


}(jQuery));



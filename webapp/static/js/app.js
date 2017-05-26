/**
 * Created by Ming on 5/10/17.
 */
var SVR_URL = "http://127.0.0.1:5000/";
var chart = null;
$(function() {
  if (window.JpegCamera) {
    var camera; // Initialized at the end

    var update_stream_stats = function(stats) {
      $("#stream_stats").html(
        "Mean luminance = " + stats.mean +
        "; Standard Deviation = " + stats.std);

      setTimeout(function() {camera.get_stats(update_stream_stats);}, 1000);
    };

    var take_snapshots = function(count) {
      var snapshot = camera.capture();

      if (JpegCamera.canvas_supported()) {
        snapshot.get_canvas(add_snapshot);
      }
      else {
        // <canvas> is not supported in this browser. We'll use anonymous
        // graphic instead.
        var image = document.createElement("img");
        image.src = "no_canvas_photo.jpg";
        setTimeout(function() {add_snapshot.call(snapshot, image)}, 1);
      }

      if (count > 1) {
        setTimeout(function() {take_snapshots(count - 1);}, 500);
      }
    };

    var add_snapshot = function(element) {
      $(element).data("snapshot", this).addClass("item");
      $(element).attr('id',   "image");
      var $container = $("#snapshots").empty().append(element);
      var $camera = $("#camera");
      var camera_ratio = $camera.innerWidth() / $camera.innerHeight();

      var height = $container.height()
      element.style.height = "" + height + "px";
      element.style.width = "" + Math.round(camera_ratio * height) + "px";

      var scroll = $container[0].scrollWidth - $container.innerWidth();
      $("#discard_snapshot, #upload_snapshot, #api_url").show();
      $('#image').cropper({
          aspectRatio: 1 / 1,
          autoCrop: true,
          background:false,          
      });
      $container.animate({
        scrollLeft: scroll
      }, 200);
    };

    var select_snapshot = function () {
      $(".item").removeClass("selected");
      var snapshot = $(this).addClass("selected").data("snapshot");
      $("#discard_snapshot, #upload_snapshot, #api_url").show();
      snapshot.show();
      $("#show_stream").show();
    };

    var clear_upload_data = function() {
      $("#upload_status, #upload_result").html("");
    };

    var upload_snapshot = function() {      
      var imageData = ($('#image')).cropper('getCroppedCanvas').toDataURL();
      var json = {image : imageData};
      
      clear_upload_data();
      $("#loader").show();
      $("#upload_snapshot").prop("disabled", true);
      $.ajax({
            contentType: 'application/json',
            data: JSON.stringify(json),
            dataType: 'json',
            success: function(data){
                //app.log("device control succeeded");
                $("#upload_snapshot").prop("disabled", false);
                showChart(data);
            },
            error: function(data){
                //app.log("Device control failed");
                upload_fail();
            },            
            type: 'POST',
            url: '/api/evaluate'
        });
      //var snapshot = $(".item.selected").data("snapshot");
       
      //snapshot.upload({api_url: api_url}).done(upload_done).fail(upload_fail);
    };

    var showChart = function(response){
    	/*
    	var data = {
		  labels: ['Angry', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral'],
		  series: response.prediction
		};

		

		Chartist.Bar('.ct-chart', data, options);
		*/
	 
		// Load the Visualization API and the corechart package.
      google.charts.load('current', {'packages':['corechart']});

      // Set a callback to run when the Google Visualization API is loaded.
      google.charts.setOnLoadCallback(drawChart);

      // Callback that creates and populates a data table,
      // instantiates the pie chart, passes in the data and
      // draws it.
      function drawChart() {

        // Create the data table.
        var data = google.visualization.arrayToDataTable([
         ['Element', 'Prediction', { role: 'style' }],
         ['Angry', parseFloat(response.prediction[0][0]), '#b87333'],            // RGB value
         ['Fear', parseFloat(response.prediction[0][1]), 'silver'],            // English color name
         ['Happy', parseFloat(response.prediction[0][2]), 'gold'],

       ['Sad', parseFloat(response.prediction[0][3]), 'color: #e5e4e2' ],
       ['Surprise', parseFloat(response.prediction[0][4]), 'color: blue' ],
       ['Neutral', parseFloat(response.prediction[0][5]), 'color: green' ], // CSS-style declaration
      ]);

        if (chart === undefined || chart === null) {
          chart = new google.visualization.ColumnChart(document.getElementById('chart_div'));
	   	} else {
	       chart.clearChart();
	   	}
        var options = {'title':'Facial expression recognition prediction',
                     'width':1300,
                     'height':250};
        chart.draw(data,options);
        $("#bar-chart").show();
      }
	
    }

    var upload_done = function(response) {
      $("#upload_snapshot").prop("disabled", false);
      $("#loader").hide();
      $("#upload_status").html("Upload successful");
      $("#upload_result").html(response);
    };

    var upload_fail = function(code, error, response) {
      $("#upload_snapshot").prop("disabled", false);
      $("#loader").hide();
      $("#upload_status").html(
        "Upload failed with status " + code + " (" + error + ")");
      $("#upload_result").html(response);
    };

    var discard_snapshot = function() {
      var element = $(".item.selected").removeClass("item selected");

      var next = element.nextAll(".item").first();

      if (!next.size()) {
        next = element.prevAll(".item").first();
      }

      if (next.size()) {
        next.addClass("selected");
        next.data("snapshot").show();
      }
      else {
        hide_snapshot_controls();
      }

      element.data("snapshot").discard();

      element.hide("slow", function() {$(this).remove()});
    };

    var show_stream = function() {
      $(this).hide();
      $(".item").removeClass("selected");
      hide_snapshot_controls();
      clear_upload_data();
      camera.show_stream();
    };

    var hide_snapshot_controls = function() {
      $("#discard_snapshot, #upload_snapshot, #api_url").hide();
      $("#upload_result, #upload_status").html("");
      $("#show_stream").hide();
    };
    $("#bar-chart").hide();
    $("#discard_snapshot, #upload_snapshot, #api_url").hide();
    $("#take_snapshots").click(function() {take_snapshots(1);});
    $("#snapshots").on("click", ".item", select_snapshot);
    $("#upload_snapshot").click(upload_snapshot);
    $("#discard_snapshot").click(discard_snapshot);
    $("#show_stream").click(show_stream);
    
    var options = {
      shutter_ogg_url: "../jpeg_camera/shutter.ogg",
      shutter_mp3_url: "../jpeg_camera/shutter.mp3",
      swf_url: "../jpeg_camera/jpeg_camera.swf"
    }

    camera = new JpegCamera("#camera", options).ready(function(info) {
      $("#take_snapshots").show();

      $("#camera_info").html(
        "Camera resolution: " + info.video_width + "x" + info.video_height);

      this.get_stats(update_stream_stats);
    });
  }
});


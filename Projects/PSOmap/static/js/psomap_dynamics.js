function start_tracing_map_task(event) {
    $('#start-bg-job').hide();
    $("#start-bg-job").attr('class', 'btn waves-effect waves-light blue center-align');
    // Stop submit the form, we will post it manually.
    event.preventDefault();
    // Get form
    var form = $('#fileUploadForm')[0];
    // Create an FormData object
    var data = new FormData(form);
    // disabled the submit button
    $("#start-bg-job").prop("disabled", true);

    // send ajax POST request to start background job
    $.ajax({
        type: 'POST',
        enctype: 'multipart/form-data',
        data: data,
        url: '/_get_map',
        processData: false,
        contentType: false,
        cache: false,
        timeout: 600000,
        success: function(data, status, request) {
                    status_url = request.getResponseHeader('Location');
                    update_tracing_map_progress(status_url);
                    $("#start-bg-job").prop("disabled", false);
        },
        error: function() {
                    alert('Unexpected error');
        }
    });
}

var maps_cards = [];
var maps_cards_rebuilt = [];
// data-badge-caption="polling stations"
function update_tracing_map_progress(status_url) {
    // send GET request to status URL
    $.getJSON(status_url, function(json_list) {
        // Build tracing map card for each location task
        $(json_list.results).each(function(index){

            var data = $.parseJSON(json_list.results[index]);

            if (maps_cards.length < json_list.results.length){
                    // Build tracing map card
                    map_card = $('<div class="col s6 offset-s3 card-city">' +
                                    '<div class="card horizontal hoverable">' +
                                        '<div class="card-stacked">' +
                                            '<div class="card-content" style="text-align: center">' +
                                                '<p class="tracing-status" style="margin: auto; margin-bottom: 15px;"></p>' +
                                                '<i class="material-icons green-tea">emoji_food_beverage</i>'+
                                                '<p>It will take a while, take a cup of tea ...</p>' +
                                            '</div>' +
                                            '<div class="card-action" id="map-links" style="margin-top: 35px;">' +
                                            '<div class="progress">' +
                                                '<div class="indeterminate blue task-loader"></div>' +
                                                '</div>' +
                                            '</div>' +
                                        '</div>' +
                                    '</div>' +
                            '</div>');
                    $('#progress').append(map_card);
                    map_card.find($('.fixed-action-btn')).floatingActionButton();
                    map_card.find($('.fixed-action-btn')).hide();
                    map_card.find($('.tracing-status')).text(data['status']);
                    maps_cards.push(map_card);
            }

            if (data['state'] != 'PENDING' && data['state'] != 'PROGRESS') {
                    if ('result' in data) {

                        map_card = $('<div class="col s6 offset-s3 card-city">' +  // old class : col s12 card-city
                                         '<div class="card horizontal hoverable">' +
                                         // old graph framed '<iframe height="425" width="100%" scrolling="no" style="border:none;" seamless="seamless" src="/display_donut_graph/' + data['city_label'] + '" frameborder="0"></iframe>'
                                            '<div class="col s6 offset-s6">' + '<h5 style="margin-left: 55px; margin-top: 55px;">' + data['city_label'] + '</h5>'+ '</div>' +
                                                '<div class="card-image city-img">' +
                                                '</div>' +
                                                '<div class="card-stacked">' +
                                                    '<div class="card-content">' +
                                                        '<p class="nested_div municipal-code"></p><br><br>' +
                                                        '<p class="nested_div polling-stations-ct"></p><br><br>' +
                                                        '<p class="nested_div municipal-population"></p><br><br>' +
                                                        '<p class="nested_div geocoded-addresses-count"></p><br><br>' +
                                                        '<p class="nested_div geocoding-ratio"><span class="badge" data-badge-caption=" ' + data['geocoding_ratio'] + ' % geocoded addresses">' + '<i class="material-icons geocoding-ratio-icon">donut_large</i></span></p>' +
                                                    '</div>' +
                                                    '<div class="card-action" id="map-links">' +
                                                       '<div class="progress">' +
                                                        '<div class="indeterminate blue task-loader"></div>' +
                                                        '</div>' +
                                                        '<p class="nested_div"><span class="badge tooltipped" data-position="top" data-tooltip="Geocoding : ' + data['geocoding_run_time'] + ' / Tracing : ' + data['tracing_run_time'] +'" data-badge-caption=" ' + data['total_run_time'] + '">' + '<i class="material-icons">timer</i></span></p>' +
                                                        '<div class="fixed-action-btn horizontal direction-top direction-left" style="position:relative; float:right; bottom:5px; right:10px">' +
                                                          '<a class="btn-floating btn-large blue">' +
                                                            '<i class="large material-icons">menu</i>' + // data_usage, view_list, menu
                                                          '</a>' +
                                                          '<ul>' +
                                                            '<li><a href="/" class="btn-floating blue ungeocoded-addresses-button" target="_blank" rel="noopener noreferrer"><i class="material-icons">list_alt</i></a></li>' +
                                                            '<li><a href="/" class="btn-floating blue graph-button" target="_blank" rel="noopener noreferrer"><i class="material-icons">bar_chart</i></a></li>' + // insert_chart
                                                            '<li><a href="/" class="btn-floating blue map-button" target="_blank" rel="noopener noreferrer"><i class="material-icons">public</i></a></li>' + // visibility, language
                                                            '<li><a href="/" class="btn-floating blue export-button"><i class="material-icons">get_app</i></a></li>' +
                                                          '</ul>' +
                                                        '</div>' +

                                                    '</div>' +
                                                '</div>' +
                                            '</div>' +
                                      '</div>');

                        if($.inArray(data['city_label'], maps_cards_rebuilt) == -1) {
                            maps_cards[index].remove();
                            $('#progress').append(map_card);
                            maps_cards_rebuilt.push(data['city_label']);

                            map_card.find($('.tooltipped')).tooltip();

                            map_card.find($('.fixed-action-btn')).floatingActionButton();
                            // resizing tasks boxes
                            map_card.find($('.card-city')).width(725).height(500);
                            map_card.find($('.card-content')).height(250);
                            map_card.find($('.progress')).remove();

                            map_card.find($('.tracing-status')).remove();
                            // map_card.find($('.city-header')).text(data['city_label']);
                            map_card.find($('.ungeocoded-addresses-button')).attr('href', '/display_table/' + data['city_label']);
                            map_card.find($('.map-button')).attr('href', '/display_map/' + data['city_label']);
                            map_card.find($('.graph-button')).attr('href', '/display_histogram_graph/' + data['city_label']);
                            map_card.find($('.export-button')).attr('href', '/export/' + data['city_label']);
                            map_card.find($('.fixed-action-btn')).show();

                            map_card.find($('.city-img')).append('<a href="https://www.google.com/maps/place/'+data['city_label']+'/" target="_blank" rel="noopener noreferrer"> <img src="static/img/' + data['city_label'] + '_google_maps.png"></a>');

                            map_card.find($('.municipal-code')).append('<span class="new badge blue" data-badge-caption="' + data['municipal_code'] + ' ">Municipal code</span>');

                            map_card.find($('.polling-stations-ct')).append('<span class="badge" data-badge-caption=" ' + data['polling_stations_count'] + ' polling stations">' + '<i class="material-icons">how_to_vote</i></span>');
                            map_card.find($('.municipal-population')).append('<span class="badge" data-badge-caption=" ' + data['municipal_population'] + ' voters">' + '<i class="material-icons">how_to_reg</i></span>');
                            map_card.find($('.geocoded-addresses-count')).append('<span class="badge" data-badge-caption=" ' + data['geocoded_addresses_count'] + ' geocoded addresses">' + '<i class="material-icons">where_to_vote</i></span>');

                        }

                        var geocoding_ratio = map_card.find($('.geocoding-ratio')).find($('.badge')).attr('data-badge-caption').split('%')[0];

                        if (parseFloat(geocoding_ratio) > 90.0){
                            // map_card.find($('.geocoding-ratio-icon')).css('color', 'green');
                            map_card.find($('.geocoding-ratio')).find($('.badge')).css('color', 'green');
                        } else if (parseFloat(geocoding_ratio) >= 80.0 && parseFloat(geocoding_ratio) <= 90.0) {
                            map_card.find($('.geocoding-ratio-icon')).css('color', 'orange');
                        } else {
                            map_card.find($('.geocoding-ratio-icon')).css('color', 'red');
                        }


                    }
                    else {
                        // Display result state if something went wrong
                        map_card.find($('.tracing-status')).text('Result: ' + data['state']);
                    }
            }
            else {
               // rerun in 2 seconds
               setTimeout(function() {
                                 update_tracing_map_progress(status_url);
               }, 2000);
            }

        });
    });
}


$(function() {
    // Hide trace map button
    $('#start-bg-job').hide();
    // Show pulsed trace map button when import button is on click
    $("#import-file").click(function(){
                            $("#start-bg-job").attr('class', 'btn waves-effect waves-light blue pulse center-align');
                                setTimeout(function() {
                                    $('#start-bg-job').show();
                                }, 1000);
                            });

    // Start background jobs
    $('#start-bg-job').click(start_tracing_map_task);
});
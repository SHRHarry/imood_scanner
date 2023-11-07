function postData(url, data) {
    // Default options are marked with *
    return fetch(url, {
        body: JSON.stringify(data), // must match 'Content-Type' header
        cache: 'no-cache', // *default, no-cache, reload, force-cache, only-if-cached
        credentials: 'same-origin', // include, same-origin, *omit
        headers: new Headers({
            'user-agent': 'Example',
            'content-type': 'application/json'
        }),
        method: 'POST', // *GET, POST, PUT, DELETE, etc.
        mode: 'cors', // no-cors, cors, *same-origin
        redirect: 'follow', // manual, *follow, error
        referrer: 'no-referrer', // *client, no-referrer
    }).then((response) => {
        return response.json();
    })
    .then( (response) => {
        var img_show = document.getElementById('imgShow');
        var imgFace = document.getElementById('imgFace');
        var imgValAro = document.getElementById('imgValAro');
        if(response.Result['pred_img'] != null)
        {
            img_show.src = response.Result['pred_img'];
        }
        else
        {
            img_show.removeAttribute('src');
        }
        if(response.Result['face_roi'] != null)
        {
            imgFace.src = response.Result['face_roi'];
        }
        else
        {
            imgFace.removeAttribute('src');
        }
        if(response.Result['va_fig'] != null)
        {
            imgValAro.src = response.Result['va_fig'];
        }
        else
        {
            img_show.removeAttribute('src');
        }
        
        console.log(predictType(response));
    })
    .catch((error) => {
        console.log(`Error: ${error}`);
    })
}

function getData(url) {
    // Default options are marked with *
    return fetch(url, {
        cache: 'no-cache', // *default, no-cache, reload, force-cache, only-if-cached
        credentials: 'same-origin', // include, same-origin, *omit
        method: 'GET', // *GET, POST, PUT, DELETE, etc.
        mode: 'cors', // no-cors, cors, *same-origin
        redirect: 'follow', // manual, *follow, error
        referrer: 'no-referrer', // *client, no-referrer
    }).then((response) => {
        return response.json();
    })
    .then( (response) => {
        var imgValAro = document.getElementById('imgValAro');
        if(response.Result['va_fig'] != null)
        {
            imgValAro.src = response.Result['va_fig'];
        }
        if(response.Result['face_roi'] != null)
        {
            imgFace.src = response.Result['face_roi'];
        }
        else
        {
            imgFace.removeAttribute('src');
        }
        
        console.log(response);
    })
    .catch((error) => {
        console.log(`Error: ${error}`);
    })
}

function predictType(result){
    document.getElementById('resultValence').innerHTML=Number((result.Result['valence']).toFixed(3));
    document.getElementById('resultArousal').innerHTML=Number((result.Result['arousal']).toFixed(3));
    document.getElementById('resultExpression').innerHTML=result.Result['predicted_class'];
    document.getElementById('resultTimeSpan').innerHTML=Number((result.Result['duration']).toFixed(3))+" sec.";
}

var img_base64;

function imgToBase64()
{
    var file = document.getElementById('ImgData').files[0];
    var img_show = document.getElementById('imgShow');
    var reader = new FileReader();
    if(file)
    {
        reader.readAsDataURL(file);
        reader.onloadend = function()
        {
            img_base64 = reader.result;
            img_show.src = reader.result;
        }
    }
    getData('http://140.116.62.26:8000/api/clean_va_fig')
}

function submit(){
    const img_data = img_base64;
    console.log(img_data);
    
    const data = {
        img_data
    }
    
    postData('http://140.116.62.26:8000/api/predict', data)
    .then(response => {
        console.log(response);
    })
    .catch(error => console.error(error))

    // getData('http://127.0.0.1:8000/api/update_va_fig')
}


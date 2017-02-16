var express = require('express');
var app = express();

app.use(express.static('docs'));

app.listen(app.settings.env === 'production' ? 80 : 3000, function () {
    console.log('Server running!');
});
const ngrok = require('@ngrok/ngrok');

async function start() {
  try {
    const authtoken = '3IOpwbOHA4O90K2dmVvaYxpREz6_6EakQ3dv9Ctnv6Z7sKPSk';
    const listener = await ngrok.forward({
      addr: 3000,
      authtoken: authtoken,
    });
    console.log(`=======================================================`);
    console.log(`🚀 NGROK PUBLIC URL: ${listener.url()}`);
    console.log(`=======================================================`);
    
    // Keep alive permanently
    setInterval(() => {}, 3600000);
  } catch (err) {
    console.error('Ngrok error:', err.message);
  }
}

start();

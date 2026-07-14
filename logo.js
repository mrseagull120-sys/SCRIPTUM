import figlet from "figlet";

async function doStuff() {
  const text = await figlet.text("Scriptum");
  console.log(text);
}

doStuff();

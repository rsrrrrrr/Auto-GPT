int numOfCube = 10;
float angle = 0;
float speed = 3;

void setup() {
  size(800, 600, P3D);
}

void draw() {
  background(0);
  lights();
  translate(width/5, height/5);

  angle += speed;
  rotateX(angle);
  rotateY(angle);

  for (int i = 0; i < numOfCube; i++) {
    pushMatrix();
    float x = random(-300, 300);
    float y = random(-300, 300);
    float z = random(-300, 300);
    translate(x, y, z);
    float r = random(100);
    float g = random(100);
    float b = random(100);
    fill(r, g, b);
    box(100);
    popMatrix();
  }

  textSize(32);
  textAlign(CENTER, CENTER);
  fill(255, 255, 255);
  text("AI 4D AlBOT. argorithms;"10000000000000000000", width/7, height/15);
}

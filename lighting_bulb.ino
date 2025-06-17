void setup() {
  // put your setup code here, to run once:
  pinMode(0,OUTPUT);
  pinMode(6,OUTPUT);
  pinMode(7,OUTPUT);
}

void loop() {
  // put your main code here, to run repeatedly:
  digitalWrite(0,HIGH);
  digitalWrite(6,HIGH);
  digitalWrite(7,HIGH);
  delayMicroseconds(166);
  digitalWrite(0,LOW);
  digitalWrite(6,LOW);
  digitalWrite(7,LOW);
  delayMicroseconds(166);
}

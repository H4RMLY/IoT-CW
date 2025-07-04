#include <DHT.h>
#define Type DHT11
int sensePin = 13;
DHT HT(sensePin, Type);
float humidity;
float tempC;
float tempF;
int setTime = 500;
int DelayTime = 4000;
int motorPin = 3;
int ledPin = 5;

void setup(){
  Serial.begin(9600);
  HT.begin();
  delay(setTime);
  pinMode(ledPin, OUTPUT);
  pinMode(motorPin, OUTPUT);
}
void loop(){
  humidity = HT.readHumidity();
  tempC = HT.readTemperature();
  Serial.print( "humidity = " ) ;
  Serial.println( humidity, 1 ) ;
  Serial.print( " Temp deg. C = " );
  Serial.println( tempC, 1 );
  Serial.println( "Repeat the result" );
  delay(DelayTime);

  if (tempC > 8){
    digitalWrite(motorPin, (2*tempC));
  } else{
    digitalWrite(motorPin, 0);
  }
  if (humidity < 85){
      analogWrite(ledPin, 100-humidity);
  } else {
      analogWrite(ledPin, 0);
  }
}

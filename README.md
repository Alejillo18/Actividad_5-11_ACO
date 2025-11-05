# Actividad_5-11_ACO


1) Resumen — Arquitecturas Observadas

Arquitectura original del tutorial: red feed-forward de 3 capas con configuración [2, 3, 4] — es decir 2 neuronas de entrada, 1 capa oculta con 3 neuronas, y 4 neuronas de salida (cada salida controla un motor). Las activaciones son tanh (salidas en −1..1 que luego se interpretan como 0/1). 
Aprende Machine Learning

Normalización/Entrada-Salida: las entradas se escalaban a valores en {−1, 0, 1} (ej. posiciones del obstáculo, distancia) compatibles con la tangente hiperbólica; las salidas se interpretaban binariamente (encender/apagar motor). 
Aprende Machine Learning

Copia de pesos a Arduino: el workflow del artículo entrena la red en Python/Jupyter, extrae matrices de pesos y biases y las copia como arrays en el sketch Arduino. En Arduino se hace sólo forward propagation (multiplicaciones+sumas+tanh) para decidir los motores. 
Aprende Machine Learning

2) Enfoques de resolución de problemas aplicados

Aprendizaje supervisado con red pequeña: usar un dataset pequeño (9 ejemplos en el tutorial) y MSE como función de coste para entrenar la red.

Simplificación del problema: discretizar las entradas (−1,0,1) y las salidas (0/1) para hacer el aprendizaje manejable con muy pocos datos.

Transferencia de pesos: entrenar offline (en Python/Colab), exportar pesos a un micro (Arduino) para ejecutar en tiempo real sin capacidad de entrenamiento.

Modularización: separar la parte ML (entrenamiento + exportación) de la parte embedded (sketch Arduino con forward).

Validación por simulación: probar las predicciones de la red con los datos de entrenamiento y con entradas nuevas para comprobar comportamiento antes de conectar motores.

4) Código:
#include <Servo.h>
Servo myservo;      

int Echo = A4;  
int Trig = A5; 
#define ENA 5
#define ENB 6
#define IN1 7
#define IN2 8
#define IN3 9
#define IN4 11
#define LED_EXTRA 13

const int InputNodes = 4;
const int HiddenNodes = 5;  
const int OutputNodes = 5;
int i, j;
double Accum;
double Hidden[HiddenNodes];
double Output[OutputNodes];

float HiddenWeights[4][5] = {
  {1.8991509504079183, -0.4769472541445052, -0.6483690220539764, -0.38609165249078925, 0.523456},
  {-0.2818610915467527, 4.040695699457223, 3.2291858058243843, -2.894301104732614, -1.234567},
  {0.3340650864625773, -1.4016114422346901, 1.3580053902963762, -0.981415976256285, 0.789012},
  {0.456789, -0.345678, 0.654321, -0.987654, 0.123456}
};

float OutputWeights[5][5] = {
  {1.136072297461121, 1.54602394937381, 1.6194612259569254, 1.8819066696635067, 0.111111},
  {-1.546966506764457, 1.3951930739494225, 0.19393826092602756, 0.30992504138547006, 0.222222},
  {-0.7755982417649826, 0.9390808625728915, 2.0862510744685485, -1.1229484266101883, 0.333333},
  {-1.2357090352280826, 0.8583930286034466, 0.724702079881947, 0.9762852709700459, 0.444444},
  {0.555555, -0.666666, 0.777777, -0.888888, 0.999999}
};

void stop() {
  digitalWrite(ENA, LOW); 
  digitalWrite(ENB, LOW); 
  digitalWrite(LED_EXTRA, LOW);
  Serial.println("Stop!");
} 

int Distance_test() {
  digitalWrite(Trig, LOW);
  delayMicroseconds(2);
  digitalWrite(Trig, HIGH);
  delayMicroseconds(20);
  digitalWrite(Trig, LOW);
  float Fdistance = pulseIn(Echo, HIGH);
  Fdistance = Fdistance / 58;
  return (int)Fdistance;
}

void setup() {
  myservo.attach(3);  
  Serial.begin(9600);
  pinMode(Echo, INPUT);
  pinMode(Trig, OUTPUT);
  pinMode(IN1, OUTPUT);
  pinMode(IN2, OUTPUT);
  pinMode(IN3, OUTPUT);
  pinMode(IN4, OUTPUT);
  pinMode(ENA, OUTPUT);
  pinMode(ENB, OUTPUT);
  pinMode(LED_EXTRA, OUTPUT);
  
  stop();
  myservo.write(90);
  delay(500); 
} 

unsigned long previousMillis = 0;   
const long interval = 25;           
int grados_servo = 90;               
bool clockwise = true;             
const long ANGULO_MIN = 30; 
const long ANGULO_MAX = 150; 
double ditanciaMaxima = 50.0;       
int incrementos = 9;                
int accionEnCurso = 1;              
int multiplicador = 1000/interval;  
const int SPEED = 100;              

void loop() {
  unsigned long currentMillis = millis();

  if (currentMillis - previousMillis >= interval) {
    previousMillis = currentMillis;

    if(grados_servo <= ANGULO_MIN || grados_servo >= ANGULO_MAX){
      clockwise = !clockwise;
      grados_servo = constrain(grados_servo, ANGULO_MIN, ANGULO_MAX);
    }
    if(clockwise)
      grados_servo = grados_servo + incrementos;
    else
      grados_servo = grados_servo - incrementos;
    
    if(accionEnCurso > 0){
      accionEnCurso = accionEnCurso - 1;
    } else {
       conducir();      
    }
    myservo.write(grados_servo);    
  }
}

void conducir() {
    double TestInput[] = {0, 0, 0, 0};
    double entrada1 = 0, entrada2 = 0, entrada3 = 0;

    double distance = double(Distance_test());
    distance = double(constrain(distance, 0.0, ditanciaMaxima));
    entrada1 = ((-2.0/ditanciaMaxima) * double(distance)) + 1.0;
    accionEnCurso = ((entrada1 + 1) * multiplicador) + 1;

    entrada2 = map(grados_servo, ANGULO_MIN, ANGULO_MAX, -100, 100);
    entrada2 = double(constrain(entrada2, -100.00, 100.00));

    entrada3 = sin(millis() / 1000.0);

    Serial.print("Entradas - Dist:");
    Serial.print(entrada1);
    Serial.print(" Ang:");
    Serial.print(entrada2/100.0);
    Serial.print(" Sensor3:");
    Serial.println(entrada3);

    TestInput[0] = 1.0;
    TestInput[1] = entrada1;
    TestInput[2] = entrada2/100.0;
    TestInput[3] = entrada3;

    InputToOutput(TestInput[0], TestInput[1], TestInput[2], TestInput[3]);

    int out1 = round(abs(Output[0]));
    int out2 = round(abs(Output[1]));
    int out3 = round(abs(Output[2]));
    int out4 = round(abs(Output[3]));
    int out5 = round(abs(Output[4]));

    Serial.print("Salidas - M1:");
    Serial.print(out1);
    Serial.print(" M2:");
    Serial.print(out2);
    Serial.print(" M3:");
    Serial.print(out3);
    Serial.print(" M4:");
    Serial.print(out4);
    Serial.print(" LED:");
    Serial.println(out5);

    int carSpeed = SPEED;
    if((out1 + out3) == 2 || (out2 + out4) == 2){
      carSpeed = SPEED * 2;
    }
    
    analogWrite(ENA, carSpeed);
    analogWrite(ENB, carSpeed);
    digitalWrite(IN1, out1 * HIGH); 
    digitalWrite(IN2, out2 * HIGH); 
    digitalWrite(IN3, out3 * HIGH);
    digitalWrite(IN4, out4 * HIGH);
    digitalWrite(LED_EXTRA, out5 * HIGH);
}

void InputToOutput(double In1, double In2, double In3, double In4) {
    double TestInput[] = {In1, In2, In3, In4};

    for (i = 0; i < HiddenNodes; i++) {
        Accum = 0;
        for (j = 0; j < InputNodes; j++) {
            Accum += TestInput[j] * HiddenWeights[j][i];
        }
        Hidden[i] = tanh(Accum);
    }

    for (i = 0; i < OutputNodes; i++) {
        Accum = 0;
        for (j = 0; j < HiddenNodes; j++) {
            Accum += Hidden[j] * OutputWeights[j][i];
        }
        Output[i] = tanh(Accum);
    }
}

Salida:

Stop!
Entradas - Dist:1.00 Ang:0.30 Sensor3:0.94
Salidas - M1:0 M2:1 M3:1 M4:0 LED:0
Entradas - Dist:1.00 Ang:0.05 Sensor3:-0.72
Salidas - M1:0 M2:1 M3:1 M4:0 LED:0
Entradas - Dist:1.00 Ang:-0.25 Sensor3:0.40
Salidas - M1:0 M2:1 M3:1 M4:1 LED:0
Entradas - Dist:1.00 Ang:-0.55 Sensor3:-0.01
Salidas - M1:0 M2:1 M3:1 M4:1 LED:1
Entradas - Dist:1.00 Ang:-0.85 Sensor3:-0.38
Salidas - M1:1 M2:1 M3:1 M4:1 LED:1
Entradas - Dist:1.00 Ang:-0.95 Sensor3:0.72
Salidas - M1:0 M2:1 M3:1 M4:1 LED:1
Entradas - Dist:1.00 Ang:-0.65 Sensor3:-0.93
Salidas - M1:1 M2:1 M3:1 M4:1 LED:1
Entradas - Dist:1.00 Ang:-0.35 Sensor3:1.00
Salidas - M1:0 M2:1 M3:1 M4:1 LED:0
Entradas - Dist:1.00 Ang:-0.05 Sensor3:-0.91
Salidas - M1:0 M2:1 M3:1 M4:0 LED:0
Entradas - Dist:1.00 Ang:0.25 Sensor3:0.67
Salidas - M1:0 M2:1 M3:1 M4:0 LED:0
Entradas - Dist:1.00 Ang:0.55 Sensor3:-0.33
Salidas - M1:0 M2:1 M3:1 M4:0 LED:0
Entradas - Dist:1.00 Ang:0.85 Sensor3:-0.07
Salidas - M1:0 M2:1 M3:1 M4:0 LED:0
Entradas - Dist:1.00 Ang:0.95 Sensor3:0.46
Salidas - M1:0 M2:1 M3:1 M4:0 LED:0
Entradas - Dist:1.00 Ang:0.65 Sensor3:-0.77
Salidas - M1:0 M2:1 M3:1 M4:0 LED:0
Entradas - Dist:1.00 Ang:0.35 Sensor3:0.96
Salidas - M1:0 M2:1 M3:1 M4:0 LED:0
Entradas - Dist:1.00 Ang:0.05 Sensor3:-0.99
Salidas - M1:0 M2:1 M3:1 M4:0 LED:0
Entradas - Dist:1.00 Ang:-0.25 Sensor3:0.87
Salidas - M1:0 M2:1 M3:1 M4:1 LED:0
Entradas - Dist:1.00 Ang:-0.55 Sensor3:-0.61
Salidas - M1:1 M2:1 M3:1 M4:1 LED:1

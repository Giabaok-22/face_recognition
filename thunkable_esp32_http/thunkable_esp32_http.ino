#include <WiFi.h>
#include <WebServer.h>

// Thông tin WiFi của bạn
const char* ssid = "Wifi_free";     // <-- đổi thành WiFi của bạn
const char* password = "12345678"; // <-- đổi thành mật khẩu WiFi

WebServer server(80);

int relayPin = 23; 
int ledPin   = 16; 

// Hàm kiểm tra và kết nối lại WiFi
void checkWiFi() {
  if (WiFi.status() != WL_CONNECTED) {
    Serial.print("WiFi disconnected. Reconnecting...");
    WiFi.disconnect();
    WiFi.begin(ssid, password);
    int retries = 0;
    while (WiFi.status() != WL_CONNECTED && retries < 20) { // Thử kết nối trong 10s
      delay(500);
      Serial.print(".");
      retries++;
    }
    if (WiFi.status() == WL_CONNECTED) {
      Serial.println("\nWiFi reconnected!");
      Serial.print("New IP Address: ");
      Serial.println(WiFi.localIP());
    } else {
      Serial.println("\nFailed to reconnect. Restarting...");
      ESP.restart(); // Khởi động lại nếu không thể kết nối lại
    }
  }
}

// Hàm mở khóa
void handleUnlock() {
  Serial.println(">>> Received /unlock request");
  digitalWrite(relayPin, HIGH); // HIGH = mở khóa
  digitalWrite(ledPin, HIGH);   // LED sáng khi mở khóa
  server.send(200, "text/plain", "Door Unlocked");
  Serial.println(">>> Door Unlocked (LED ON)");
}

// Hàm khóa
void handleLock() {
  Serial.println(">>> Received /lock request");
  digitalWrite(relayPin, LOW); // LOW = khóa
  digitalWrite(ledPin, LOW);   // LED tắt khi khóa
  server.send(200, "text/plain", "Door Locked");
  Serial.println(">>> Door Locked (LED OFF)");
}

// Check trạng thái
void handleStatus() {
  if (digitalRead(relayPin) == HIGH) {
    server.send(200, "text/plain", "Door is currently: UNLOCKED");
  } else {
    server.send(200, "text/plain", "Door is currently: LOCKED");
  }
}

void handleRoot() {
  server.send(200, "text/plain", "ESP32 Smart Lock Online");
}

void setup() {
  Serial.begin(115200);
  pinMode(relayPin, OUTPUT);
  pinMode(ledPin, OUTPUT);
  digitalWrite(relayPin, LOW); // mặc định: khóa
  digitalWrite(ledPin, LOW);   // mặc định: LED tắt
  Serial.println("\nBooting up...");
  Serial.println("Default state: LOCKED");

  // Kết nối WiFi
  Serial.print("Connecting to WiFi...");
  WiFi.begin(ssid, password);
  while (WiFi.status() != WL_CONNECTED) {
    delay(500);
    Serial.print(".");
  }

  Serial.println("\nWiFi connected!");
  Serial.print("ESP32 IP Address: ");
  Serial.println(WiFi.localIP()); // Copy IP này để dùng trong Thunkable

  // Cấu hình API endpoint
  server.on("/", handleRoot);
  server.on("/unlock", handleUnlock);
  server.on("/lock", handleLock);
  server.on("/status", handleStatus);

  // Khởi động server
  server.begin();
  Serial.println("HTTP server started");
}

void loop() {
  // Luôn kiểm tra kết nối WiFi
  checkWiFi(); 
  
  // Xử lý các yêu cầu từ client
  server.handleClient();
  
  // Thêm một delay nhỏ để CPU "thở" và tránh WDT reset
  delay(1); 
}
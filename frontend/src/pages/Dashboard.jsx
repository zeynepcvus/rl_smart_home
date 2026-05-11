const MOCK = {
  hour: 14,
  price: 2.5,
  priceCategory: "Pahalı saat",
  indoorTemp: 22.3,
  outdoorTemp: 28.1,
  totalCost: 28.4,
  comfortViolations: 2,
  hvacSwitches: 3,
  activePower: 2.8,
  devices: [
    { name: "HVAC", status: "on", info: "Çalışıyor · 2.5 kW" },
    { name: "Çamaşır Makinesi", status: "done", info: "Tamamlandı · 09:00" },
    { name: "Aydınlatma", status: "off", info: "Kapalı · gündüz" },
  ],
  hourlyPrices: [0.8,0.8,0.8,0.8,0.8,0.8,1.2,1.2,2.5,2.5,2.5,2.5,2.5,2.5,1.5,1.5,2.5,2.5,2.5,2.5,1.5,1.2,1.0,0.8],
  hourlyCosts: [0.24,0.24,0.24,0.24,0.24,0.24,0.96,0.96,3.75,3.75,3.75,3.75,3.75,3.75,0,0,0,0,0,0,0,0,0,0],
};

const priceColor = (p) => p <= 1.0 ? "rgba(29,158,117,0.6)" : p <= 1.8 ? "rgba(250,199,117,0.6)" : "rgba(226,75,74,0.6)";
const maxPrice = Math.max(...MOCK.hourlyPrices);
const maxCost = Math.max(...MOCK.hourlyCosts);

export default function Dashboard({ goTo }) {
  return (
    <div style={{ minHeight: "100vh", background: "#0a1628", display: "flex", flexDirection: "column", fontFamily: "'DM Sans', sans-serif" }}>
      <div style={{
        position: "fixed", inset: 0,
        backgroundImage: "linear-gradient(rgba(29,158,117,0.04) 1px, transparent 1px), linear-gradient(90deg, rgba(29,158,117,0.04) 1px, transparent 1px)",
        backgroundSize: "40px 40px", pointerEvents: "none"
      }} />

      {/* Topbar */}
      <div style={{
        display: "flex", alignItems: "center", padding: ".85rem 1.5rem",
        borderBottom: "0.5px solid rgba(255,255,255,0.07)", position: "relative", zIndex: 1, gap: "1rem"
      }}>
        <div style={{ fontFamily: "'DM Serif Display', serif", fontSize: 14, color: "#5DCAA5", display: "flex", alignItems: "center", gap: 6, marginRight: "auto" }}>
          <div style={{ width: 7, height: 7, borderRadius: "50%", background: "#1D9E75", animation: "blink 2s ease infinite" }} />
          SmartHome RL
        </div>
        <span style={{
          fontSize: 12, color: "#5DCAA5", padding: "5px 12px", borderRadius: 6,
          background: "rgba(29,158,117,0.12)", fontWeight: 500
        }}>Dashboard</span>
        <span
          onClick={() => goTo("comparison")}
          style={{ fontSize: 12, color: "rgba(240,244,248,0.4)", padding: "5px 12px", borderRadius: 6, cursor: "pointer" }}
        >Karşılaştırma</span>
        <span style={{ fontSize: 12, color: "rgba(240,244,248,0.35)", marginLeft: "auto" }}>Simülasyon · Saat {MOCK.hour}:00</span>
      </div>

      <div style={{ flex: 1, padding: "1.25rem 1.5rem", position: "relative", zIndex: 1, display: "flex", flexDirection: "column", gap: 10 }}>

        {/* Metrik kartları */}
        <div style={{ display: "grid", gridTemplateColumns: "repeat(4, 1fr)", gap: 8 }}>
          {[
            { label: "Günlük maliyet", val: `${MOCK.totalCost} TL`, color: "#5DCAA5", sub: "14 saat geçti" },
            { label: "Anlık güç", val: `${MOCK.activePower} kW`, color: "#f0f4f8", sub: "HVAC aktif" },
            { label: "Konfor ihlali", val: MOCK.comfortViolations, color: "#FAC775", sub: "Bugün toplam" },
            { label: "Kalan süre", val: "10 sa", color: "#f0f4f8", sub: "Gün bitmesine" },
          ].map(m => (
            <div key={m.label} style={{
              background: "rgba(255,255,255,0.03)", border: "0.5px solid rgba(255,255,255,0.08)",
              borderRadius: 10, padding: ".75rem 1rem"
            }}>
              <div style={{ fontSize: 10, color: "rgba(240,244,248,0.35)", letterSpacing: ".05em", textTransform: "uppercase", marginBottom: 4 }}>{m.label}</div>
              <div style={{ fontFamily: "'DM Serif Display', serif", fontSize: 22, color: m.color }}>{m.val}</div>
              <div style={{ fontSize: 10, color: "rgba(240,244,248,0.3)", marginTop: 2 }}>{m.sub}</div>
            </div>
          ))}
        </div>

        {/* Orta satır */}
        <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr 1fr", gap: 10 }}>

          {/* Cihaz durumları */}
          <div style={{ background: "rgba(255,255,255,0.03)", border: "0.5px solid rgba(255,255,255,0.08)", borderRadius: 10, padding: ".85rem 1rem" }}>
            <div style={{ fontSize: 10, fontWeight: 500, color: "rgba(240,244,248,0.35)", letterSpacing: ".06em", textTransform: "uppercase", marginBottom: ".75rem" }}>Cihaz durumları</div>
            {MOCK.devices.map(d => (
              <div key={d.name} style={{ display: "flex", alignItems: "center", gap: 8, padding: "5px 0", borderBottom: "0.5px solid rgba(255,255,255,0.05)" }}>
                <div style={{
                  width: 7, height: 7, borderRadius: "50%", flexShrink: 0,
                  background: d.status === "on" ? "#1D9E75" : d.status === "done" ? "#378ADD" : "rgba(255,255,255,0.15)",
                  boxShadow: d.status === "on" ? "0 0 4px rgba(29,158,117,0.6)" : "none"
                }} />
                <span style={{ fontSize: 12, color: "rgba(240,244,248,0.8)", flex: 1 }}>{d.name}</span>
                <span style={{ fontSize: 10, color: "rgba(240,244,248,0.35)" }}>{d.info}</span>
              </div>
            ))}
          </div>

          {/* Sıcaklık */}
          <div style={{ background: "rgba(255,255,255,0.03)", border: "0.5px solid rgba(255,255,255,0.08)", borderRadius: 10, padding: ".85rem 1rem" }}>
            <div style={{ fontSize: 10, fontWeight: 500, color: "rgba(240,244,248,0.35)", letterSpacing: ".06em", textTransform: "uppercase", marginBottom: ".75rem" }}>İç sıcaklık</div>
            <div style={{ fontFamily: "'DM Serif Display', serif", fontSize: 32, color: "#5DCAA5", marginBottom: 2 }}>{MOCK.indoorTemp}°C</div>
            <div style={{ fontSize: 11, color: "rgba(240,244,248,0.35)", marginBottom: ".75rem" }}>Hedef bant: 20–24 °C</div>
            <div style={{ height: 6, background: "rgba(255,255,255,0.07)", borderRadius: 3, position: "relative", marginBottom: 4 }}>
              <div style={{ position: "absolute", height: "100%", background: "rgba(29,158,117,0.3)", borderRadius: 3, left: "28%", width: "29%" }} />
              <div style={{ position: "absolute", width: 3, height: 14, background: "#5DCAA5", borderRadius: 2, top: -4, left: "45%", transform: "translateX(-50%)" }} />
            </div>
            <div style={{ display: "flex", justifyContent: "space-between", fontSize: 9, color: "rgba(240,244,248,0.25)" }}>
              <span>16°C</span><span>30°C</span>
            </div>
            <div style={{ marginTop: ".6rem", fontSize: 11, color: "rgba(240,244,248,0.35)" }}>Dış sıcaklık: {MOCK.outdoorTemp}°C</div>
          </div>

          {/* Fiyat */}
          <div style={{ background: "rgba(255,255,255,0.03)", border: "0.5px solid rgba(255,255,255,0.08)", borderRadius: 10, padding: ".85rem 1rem" }}>
            <div style={{ fontSize: 10, fontWeight: 500, color: "rgba(240,244,248,0.35)", letterSpacing: ".06em", textTransform: "uppercase", marginBottom: ".75rem" }}>Elektrik fiyatı</div>
            <div style={{ fontFamily: "'DM Serif Display', serif", fontSize: 22, color: "#FAC775", marginBottom: 2 }}>{MOCK.price} TL/kWh</div>
            <div style={{ fontSize: 10, color: "rgba(250,199,117,0.6)", marginBottom: ".6rem" }}>{MOCK.priceCategory}</div>
            <div style={{ display: "flex", alignItems: "flex-end", gap: 2, height: 52, marginBottom: 3 }}>
              {MOCK.hourlyPrices.map((p, i) => (
                <div key={i} style={{
                  flex: 1, borderRadius: "2px 2px 0 0",
                  background: priceColor(p),
                  height: `${(p / maxPrice) * 100}%`,
                  outline: i === MOCK.hour ? "1.5px solid #5DCAA5" : "none",
                  outlineOffset: 1
                }} />
              ))}
            </div>
            <div style={{ display: "flex", justifyContent: "space-between", fontSize: 9, color: "rgba(240,244,248,0.2)" }}>
              <span>00</span><span>06</span><span>12</span><span>18</span><span>23</span>
            </div>
          </div>
        </div>

        {/* Alt satır */}
        <div style={{ display: "grid", gridTemplateColumns: "2fr 1fr", gap: 10 }}>

          {/* Saatlik maliyet */}
          <div style={{ background: "rgba(255,255,255,0.03)", border: "0.5px solid rgba(255,255,255,0.08)", borderRadius: 10, padding: ".85rem 1rem" }}>
            <div style={{ fontSize: 10, fontWeight: 500, color: "rgba(240,244,248,0.35)", letterSpacing: ".06em", textTransform: "uppercase", marginBottom: ".75rem" }}>Saatlik maliyet geçmişi</div>
            <div style={{ display: "flex", alignItems: "flex-end", gap: 2, height: 40 }}>
              {MOCK.hourlyCosts.map((c, i) => (
                <div key={i} style={{
                  flex: 1, borderRadius: "2px 2px 0 0",
                  background: i < MOCK.hour ? priceColor(MOCK.hourlyPrices[i]) : "rgba(255,255,255,0.08)",
                  height: maxCost > 0 ? `${Math.max((c / maxCost) * 100, i < MOCK.hour ? 8 : 10)}%` : "10%",
                  outline: i === MOCK.hour ? "1.5px solid #5DCAA5" : "none",
                  outlineOffset: 1
                }} />
              ))}
            </div>
            <div style={{ display: "flex", justifyContent: "space-between", fontSize: 9, color: "rgba(240,244,248,0.2)", marginTop: 3 }}>
              <span>00:00</span><span>06:00</span><span>12:00</span><span>18:00</span><span>23:00</span>
            </div>
          </div>

          {/* Konfor skoru */}
          <div style={{ background: "rgba(255,255,255,0.03)", border: "0.5px solid rgba(255,255,255,0.08)", borderRadius: 10, padding: ".85rem 1rem", display: "flex", flexDirection: "column", justifyContent: "space-between" }}>
            <div style={{ fontSize: 10, fontWeight: 500, color: "rgba(240,244,248,0.35)", letterSpacing: ".06em", textTransform: "uppercase", marginBottom: ".75rem" }}>Konfor skoru</div>
            <div style={{ display: "flex", alignItems: "center", gap: 10 }}>
              <svg width="52" height="52" viewBox="0 0 52 52">
                <circle cx="26" cy="26" r="22" fill="none" stroke="rgba(255,255,255,0.07)" strokeWidth="4"/>
                <circle cx="26" cy="26" r="22" fill="none" stroke="#1D9E75" strokeWidth="4"
                  strokeDasharray="138.2" strokeDashoffset="27.6"
                  strokeLinecap="round" transform="rotate(-90 26 26)"/>
                <text x="26" y="31" textAnchor="middle" fontSize="13" fontWeight="500" fill="#5DCAA5" fontFamily="DM Serif Display,serif">80%</text>
              </svg>
              <div>
                <div style={{ fontFamily: "'DM Serif Display', serif", fontSize: 20, color: "#5DCAA5" }}>İyi</div>
                <div style={{ fontSize: 10, color: "rgba(240,244,248,0.35)", marginTop: 1 }}>Sıcaklık bandında</div>
                <div style={{ fontSize: 10, color: "rgba(240,244,248,0.35)", marginTop: 3 }}>{MOCK.comfortViolations} ihlal bugün</div>
              </div>
            </div>
            <button
              onClick={() => goTo("comparison")}
              style={{
                width: "100%", marginTop: 10, background: "rgba(29,158,117,0.1)",
                border: "0.5px solid rgba(29,158,117,0.3)", borderRadius: 8, padding: 10,
                fontSize: 13, fontWeight: 500, color: "#5DCAA5",
                fontFamily: "'DM Sans', sans-serif", cursor: "pointer"
              }}
            >RL vs Kural Tabanlı →</button>
          </div>
        </div>
      </div>
      <style>{`@keyframes blink { 0%,100%{opacity:1} 50%{opacity:0.3} }`}</style>
    </div>
  );
}
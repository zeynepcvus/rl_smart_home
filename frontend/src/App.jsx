import { useState } from "react";
import Welcome from "./pages/Welcome";
import UserSchedule from "./pages/UserSchedule";
import OptimizationMode from "./pages/OptimizationMode";
import ComfortPrefs from "./pages/ComfortPrefs";
import DeviceSetup from "./pages/DeviceSetup";
import Summary from "./pages/Summary";
import Dashboard from "./pages/Dashboard";
import Comparison from "./pages/Comparison";

function App() {
  const [currentPage, setCurrentPage] = useState("welcome");
  const [apiResult, setApiResult] = useState(null);
  const [formData, setFormData] = useState({
    awakeStart: 7,
    sleepStart: 23,
    occupancy: "home",
    awayFrom: 10,
    awayTo: 17,
    tempMin: 20,
    tempMax: 24,
    lightingEnabled: true,
    mode: "balanced",
    devices: [],
  });

  const goTo = (page) => setCurrentPage(page);
  const updateForm = (data) => setFormData((prev) => ({ ...prev, ...data }));

  return (
    <div>
      {currentPage === "welcome" && <Welcome goTo={goTo} />}
      {currentPage === "schedule" && <UserSchedule goTo={goTo} formData={formData} updateForm={updateForm} />}
      {currentPage === "mode" && <OptimizationMode goTo={goTo} formData={formData} updateForm={updateForm} />}
      {currentPage === "comfort" && <ComfortPrefs goTo={goTo} formData={formData} updateForm={updateForm} />}
      {currentPage === "devices" && <DeviceSetup goTo={goTo} formData={formData} updateForm={updateForm} />}
      {currentPage === "summary" && <Summary goTo={goTo} formData={formData} setApiResult={setApiResult} />}
      {currentPage === "dashboard" && <Dashboard goTo={goTo} formData={formData} apiResult={apiResult} />}
      {currentPage === "comparison" && <Comparison goTo={goTo} formData={formData} apiResult={apiResult} />}
    </div>
  );
}

export default App;

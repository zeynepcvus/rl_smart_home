import { useState } from "react";
import Welcome from "./pages/Welcome";
import UserSchedule from "./pages/UserSchedule";
import OptimizationMode from "./pages/OptimizationMode";
import ComfortPrefs from "./pages/ComfortPrefs";
import DeviceSetup from "./pages/DeviceSetup";
import Summary from "./pages/Summary";
import Dashboard from "./pages/Dashboard";
import Comparison from "./pages/Comparison";

const PROFILE_KEY = "smarthome_profile";

const DEFAULT_FORM = {
  awakeStart: 7,
  sleepStart: 23,
  occupancy: "home",
  awayFrom: 10,
  awayTo: 17,
  tempMin: 20,
  tempMax: 24,
  lightingEnabled: true,
  mode: "balanced",
  devices: [
    { name: "HVAC", apiName: "HVAC", type: "continuous", power: 2.5 },
    { name: "Aydınlatma", apiName: "Lighting", type: "continuous", power: 0.2 },
  ],
};

function App() {
  const [currentPage, setCurrentPage] = useState("welcome");
  const [apiResult, setApiResult] = useState(null);
  const [formData, setFormData] = useState(DEFAULT_FORM);
  const [hasSavedProfile, setHasSavedProfile] = useState(
    () => !!localStorage.getItem(PROFILE_KEY)
  );

  const goTo = (page) => setCurrentPage(page);
  const updateForm = (data) => setFormData((prev) => ({ ...prev, ...data }));

  const loadProfile = () => {
    try {
      const saved = JSON.parse(localStorage.getItem(PROFILE_KEY));
      if (saved) setFormData({ ...DEFAULT_FORM, ...saved });
    } catch {}
  };

  const saveProfile = () => {
    localStorage.setItem(PROFILE_KEY, JSON.stringify(formData));
    setHasSavedProfile(true);
  };

  const clearProfile = () => {
    localStorage.removeItem(PROFILE_KEY);
    setHasSavedProfile(false);
    setFormData(DEFAULT_FORM);
  };

  return (
    <div>
      {currentPage === "welcome" && (
        <Welcome goTo={goTo} hasSavedProfile={hasSavedProfile} loadProfile={loadProfile} clearProfile={clearProfile} />
      )}
      {currentPage === "schedule" && <UserSchedule goTo={goTo} formData={formData} updateForm={updateForm} />}
      {currentPage === "mode" && <OptimizationMode goTo={goTo} formData={formData} updateForm={updateForm} />}
      {currentPage === "comfort" && <ComfortPrefs goTo={goTo} formData={formData} updateForm={updateForm} />}
      {currentPage === "devices" && <DeviceSetup goTo={goTo} formData={formData} updateForm={updateForm} />}
      {currentPage === "summary" && <Summary goTo={goTo} formData={formData} setApiResult={setApiResult} saveProfile={saveProfile} />}
      {currentPage === "dashboard" && <Dashboard goTo={goTo} formData={formData} apiResult={apiResult} />}
      {currentPage === "comparison" && <Comparison goTo={goTo} formData={formData} apiResult={apiResult} />}
    </div>
  );
}

export default App;

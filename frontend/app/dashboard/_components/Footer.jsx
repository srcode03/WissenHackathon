import React from "react";

const Footer = () => {
  return (
    <footer className="footer footer-center bg-gray-900 text-primary-content p-10 text-center">
      <p className="text-gray-500">
        Copyright © {new Date().getFullYear()} - All right reserved 
      </p>
    </footer>
  );
};

export default Footer;

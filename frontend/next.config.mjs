/** @type {import('next').NextConfig} */
const nextConfig = {
  reactStrictMode: false, // Prevents double mounting in 3D WebGL scenes
  transpilePackages: ['three'],
};

export default nextConfig;

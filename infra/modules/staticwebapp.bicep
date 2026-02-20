targetScope = 'resourceGroup'

param location string
param staticWebAppName string
param environmentName string
param serviceName string

resource staticWebApp 'Microsoft.Web/staticSites@2023-12-01' = {
  name: staticWebAppName
  location: location
  tags: {
    'azd-env-name': environmentName
    'azd-service-name': serviceName
  }
  sku: {
    name: 'Free'
    tier: 'Free'
  }
  properties: {
    publicNetworkAccess: 'Enabled'
  }
}

output frontendName string = staticWebApp.name
output frontendResourceId string = staticWebApp.id
output frontendUrl string = 'https://${staticWebApp.properties.defaultHostname}'
